"""
Baseline GSM8K evaluation: how good is the model *before* RL, and will RL have a signal?

Two measurements:
  greedy  -- one deterministic answer per test question -> accuracy (strict / lenient)
  sampled -- k samples per question at the RL temperature -> pass@1, pass@k, and the
             fraction of questions whose k samples have MIXED rewards. Only mixed groups
             get a non-zero GRPO advantage, so that fraction is the share of each batch
             that actually produces gradient (PicoChat's was ~0).

Run:  python -m rl.eval --n 200 --k 0          (quick greedy check)
      python -m rl.eval                        (full test greedy + 256 train questions x 8 samples)
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from rl.data import build_prompt, load_gsm8k
from rl.reward import score


def load_model(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"          # decoder-only batching: pad on the left so every row's last token is real
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
    return model, tok


def gen_config(model, do_sample, temperature, max_new_tokens):
    """Built from scratch on purpose: Qwen's shipped generation_config sets top_k=20, top_p=0.8
    and repetition_penalty=1.1. Those distort the sampling distribution, and RL needs samples
    from the model's true distribution (the log-probs we train on must be the ones we sampled from).
    top_k must be 0, not None: None falls back to the shipped top_k=20 (checked -- a TopKLogitsWarper
    appears). transformers warns "top_k not valid" for 0; that warning is harmless."""
    return GenerationConfig(
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=1.0, top_k=0, repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=model.generation_config.pad_token_id,
    )


def completion_lengths(tokens, eos_ids):
    """Tokens up to and including the first EOS; rows with no EOS were truncated (length = T)."""
    is_eos = torch.isin(tokens, torch.tensor(eos_ids, device=tokens.device))
    T = tokens.shape[1]
    idx = torch.arange(T, device=tokens.device).expand_as(tokens)
    first = torch.where(is_eos, idx, torch.full_like(idx, T)).min(dim=1).values
    truncated = first == T
    return torch.where(truncated, first, first + 1).tolist(), truncated.tolist()


@torch.no_grad()
def generate(model, tok, prompts, cfg, batch_size, num_samples=1, verbose=True):
    """-> list (len(prompts)) of lists (num_samples) of dicts {text, length, truncated}."""
    eos_ids = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
    cfg.num_return_sequences = num_samples
    per_batch = max(1, batch_size // num_samples)            # batch_size counts sequences, not prompts
    out = []
    for i in range(0, len(prompts), per_batch):
        enc = tok(prompts[i:i + per_batch], return_tensors="pt", padding=True).to(model.device)
        seqs = model.generate(**enc, generation_config=cfg)
        comp = seqs[:, enc["input_ids"].shape[1]:]           # left padding -> every prompt ends at the same column
        lengths, truncated = completion_lengths(comp, eos_ids)
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        rows = [{"text": t, "length": l, "truncated": tr} for t, l, tr in zip(texts, lengths, truncated)]
        out += [rows[j:j + num_samples] for j in range(0, len(rows), num_samples)]
        if verbose:
            print(f"  {min(i + per_batch, len(prompts))}/{len(prompts)}", flush=True)
    return out


def summarize_greedy(problems, gens):
    records, n = [], len(problems)
    for p, g in zip(problems, gens):
        s = score(g[0]["text"], p["answer"])
        records.append({**p, **g[0], **s})
    src = {k: sum(r["source"] == k for r in records) / n for k in ("hash", "boxed", "last", None)}
    summary = {
        "n": n,
        "acc_strict": sum(r["correct_strict"] for r in records) / n,
        "acc_lenient": sum(r["correct_lenient"] for r in records) / n,
        "answer_source": {str(k): v for k, v in src.items()},
        "mean_length": sum(r["length"] for r in records) / n,
        "truncated_frac": sum(r["truncated"] for r in records) / n,
    }
    return summary, records


def summarize_sampled(problems, gens, k):
    records, per_q = [], []
    for p, g in zip(problems, gens):
        c = [score(x["text"], p["answer"])["correct_strict"] for x in g]
        per_q.append(sum(c))
        records.append({"question": p["question"], "answer": p["answer"], "n_correct": sum(c),
                        "samples": [{**x, "correct": ci} for x, ci in zip(g, c)]})
    n = len(problems)
    summary = {
        "n": n, "k": k,
        "pass@1": sum(per_q) / (n * k),
        f"pass@{k}": sum(c > 0 for c in per_q) / n,
        "mixed_frac": sum(0 < c < k for c in per_q) / n,     # these groups give GRPO gradient
        "all_correct_frac": sum(c == k for c in per_q) / n,
        "all_wrong_frac": sum(c == 0 for c in per_q) / n,
        "n_correct_hist": [per_q.count(i) for i in range(k + 1)],
        "mean_length": sum(x["length"] for g in gens for x in g) / (n * k),
        "truncated_frac": sum(x["truncated"] for g in gens for x in g) / (n * k),
    }
    return summary, records


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="hash", choices=["hash", "boxed"])
    ap.add_argument("--n", type=int, default=None, help="limit greedy test questions (default: all 1,319)")
    ap.add_argument("--k", type=int, default=8, help="samples per question for the sampled eval; 0 skips it")
    ap.add_argument("--k-split", default="train", help="RL trains on train, so the signal check uses train")
    ap.add_argument("--k-n", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128, help="sequences per generate call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = args.out or os.path.join("runs", "eval", f"{args.model.split('/')[-1]}_{args.prompt}")
    os.makedirs(out, exist_ok=True)
    model, tok = load_model(args.model)
    results = {"args": vars(args)}

    test = load_gsm8k("test")[:args.n]
    print(f"[greedy] {len(test)} test questions")
    t0 = time.time()
    gens = generate(model, tok, [build_prompt(tok, p["question"], args.prompt) for p in test],
                    gen_config(model, False, None, args.max_new_tokens), args.batch_size)
    results["greedy"], recs = summarize_greedy(test, gens)
    results["greedy"]["seconds"] = round(time.time() - t0, 1)
    write_jsonl(os.path.join(out, "greedy.jsonl"), recs)
    print(json.dumps(results["greedy"], indent=2))

    if args.k > 0:
        g = torch.Generator().manual_seed(args.seed)             # fixed question subset, independent of --n
        pool = load_gsm8k(args.k_split)
        subset = [pool[i] for i in torch.randperm(len(pool), generator=g)[:args.k_n].tolist()]
        print(f"[sampled] {len(subset)} {args.k_split} questions x {args.k} samples @ T={args.temperature}")
        t0 = time.time()
        gens = generate(model, tok, [build_prompt(tok, p["question"], args.prompt) for p in subset],
                        gen_config(model, True, args.temperature, args.max_new_tokens),
                        args.batch_size, num_samples=args.k)
        results["sampled"], recs = summarize_sampled(subset, gens, args.k)
        results["sampled"]["seconds"] = round(time.time() - t0, 1)
        write_jsonl(os.path.join(out, "sampled.jsonl"), recs)
        print(json.dumps(results["sampled"], indent=2))

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {out}/summary.json")


if __name__ == "__main__":
    main()
