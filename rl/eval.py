"""
GSM8K evaluation of a model, optionally with a trained LoRA adapter.

Three measurements:
  greedy  -- one deterministic answer per test question -> accuracy (strict / lenient)
  sampled -- k samples per question at the RL temperature -> pass@1 ... pass@k (unbiased
             estimator), and the fraction of questions whose k samples have MIXED rewards
             (only mixed groups give GRPO a gradient)
  hard    -- (--hard runs/grpo/<run>/hard.jsonl) re-ask the questions a run never solved
             during training (0/8) and count how many are solved now

Run:  python -m rl.eval --n 200 --k 0                              (quick greedy check)
      python -m rl.eval --adapter runs/grpo/<run>/adapter           (a trained adapter)
      python -m rl.eval --k 64 --k-n 0 --engine vllm               (pass@1..64 on the full test)
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from rl.data import eos_ids_of, load_gsm8k
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
        eos_token_id=eos_ids_of(model.generation_config),       # includes <|im_end|> (see rl.data)
        pad_token_id=model.generation_config.pad_token_id or eos_ids_of(model.generation_config)[0],
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
        rows = [{"text": t, "length": l, "truncated": tr, "token_ids": ids[:l].tolist(),
                 "finish_reason": "length" if tr else "stop"} for t, l, tr, ids in zip(texts, lengths, truncated, comp)]
        out += [rows[j:j + num_samples] for j in range(0, len(rows), num_samples)]
        if verbose:
            print(f"  {min(i + per_batch, len(prompts))}/{len(prompts)}", flush=True)
    return out


def pass_at_k(n, c, k):
    """Chance that at least one of k tries is right, estimated from n tries with c right
    (the unbiased estimator from the Codex paper, Chen et al. 2021): 1 - C(n-c, k) / C(n, k).
    Better than "was one of the first k right": it uses all n tries, so it is less noisy.
    Examples, n=8: c=0 -> 0; c=8 -> 1; c=1, k=1 -> 1/8; c=1, k=8 -> 1; c=4, k=2 -> 1 - 6/28 = 0.786."""
    if n - c < k:
        return 1.0
    p_all_wrong = 1.0
    for i in range(k):                                    # C(n-c, k) / C(n, k) as a product
        p_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - p_all_wrong


def passk_summary(counts, n, ks=None):
    """counts: right answers per question (out of n tries each) -> {"pass@k": mean over questions}."""
    ks = ks or [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= n]
    return {f"pass@{k}": sum(pass_at_k(n, c, k) for c in counts) / len(counts) for k in ks}


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
        "passk": passk_summary(per_q, k),
        "mean_length": sum(x["length"] for g in gens for x in g) / (n * k),
        "truncated_frac": sum(x["truncated"] for g in gens for x in g) / (n * k),
    }
    return summary, records


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_hard(path):
    """Unique questions from a run's hard.jsonl (questions whose 8 samples were all wrong)."""
    seen, out = set(), []
    for line in open(path):
        r = json.loads(line)
        if r["idx"] not in seen:
            seen.add(r["idx"])
            out.append({"question": r["question"], "answer": r["answer"], "idx": r["idx"], "step": r["step"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="a saved LoRA adapter dir (e.g. runs/grpo/<run>/adapter)")
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--prompt", default="boxed", choices=["hash", "boxed", "plain"])
    ap.add_argument("--n", type=int, default=None, help="limit greedy test questions (default: all 1,319; 0 skips)")
    ap.add_argument("--k", type=int, default=8, help="samples per question for the sampled eval; 0 skips it")
    ap.add_argument("--k-split", default="test", help="test (pass@k) or train (Phase 1's signal check)")
    ap.add_argument("--k-n", type=int, default=256, help="questions for the sampled eval (0 = the whole split)")
    ap.add_argument("--hard", default=None, help="a run's hard.jsonl: re-check its never-solved questions")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128, help="HF engine: sequences per generate call")
    ap.add_argument("--vllm-mem", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", help="pinned question manifest for resumable campaign evaluation")
    ap.add_argument("--model-revision", default=None)
    args = ap.parse_args()
    if args.manifest:
        from rl.campaign import MODEL_REVISION, isolated_rng
        from rl.evaluation_io import evaluate_manifest
        from rl.engine import HFEngine, VLLMEngine
        from rl.checkpointing import file_hash
        revision = args.model_revision or MODEL_REVISION
        manifest = json.load(open(args.manifest))
        if args.out is None:
            raise ValueError("campaign evaluation requires --out")
        with isolated_rng():
            tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
            tok.padding_side = "left"
            if args.engine == "vllm":
                engine = VLLMEngine(None, tok, args.model, args.max_new_tokens, args.prompt,
                                    adapter_dir=args.adapter, gpu_memory_utilization=args.vllm_mem, revision=revision)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=torch.bfloat16).cuda().eval()
                if args.adapter:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, args.adapter).eval()
                engine = HFEngine(model, tok, args.max_new_tokens, args.prompt, args.batch_size)
        identity = {"model": args.model, "revision": revision, "prompt": args.prompt,
                    "max_new_tokens": args.max_new_tokens, "engine": args.engine,
                    "adapter_sha256": file_hash(os.path.join(args.adapter,"adapter_model.safetensors")) if args.adapter else None}
        if args.n != 0:
            evaluate_manifest(engine, manifest, os.path.join(args.out,"greedy"), 1, sampled=False, model_identity=identity)
        if args.k > 0:
            evaluate_manifest(engine, manifest, os.path.join(args.out,"sampled"), args.k, model_identity=identity)
        return

    from rl.engine import HFEngine, VLLMEngine
    torch.manual_seed(args.seed)
    name = args.model.split("/")[-1] + (f"+{os.path.basename(os.path.dirname(args.adapter.rstrip('/')))}"
                                        if args.adapter else "")
    out = args.out or os.path.join("runs", "eval", f"{name}_{args.prompt}")
    os.makedirs(out, exist_ok=True)
    if args.engine == "hf":
        model, tok = load_model(args.model)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).eval()
        engine = HFEngine(model, tok, args.max_new_tokens, args.prompt, args.batch_size)
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        engine = VLLMEngine(None, tok, args.model, args.max_new_tokens, args.prompt, adapter_dir=args.adapter,
                            gpu_memory_utilization=args.vllm_mem, seed=args.seed)
    results = {"args": vars(args)}

    if args.n != 0:
        test = load_gsm8k("test")[:args.n]
        print(f"[greedy] {len(test)} test questions")
        t0 = time.time()
        gens = [[g] for g in engine.greedy(test)]
        results["greedy"], recs = summarize_greedy(test, gens)
        results["greedy"]["seconds"] = round(time.time() - t0, 1)
        write_jsonl(os.path.join(out, "greedy.jsonl"), recs)
        print(json.dumps(results["greedy"], indent=2))

    def sampled(problems, tag):
        t0 = time.time()
        ro = engine.sample(problems, args.k, args.temperature)
        texts, lengths, trunc = ro["texts"], ro["lengths"].tolist(), ro["truncated"].tolist()
        gens = [[{"text": texts[i * args.k + j], "length": lengths[i * args.k + j],
                  "truncated": trunc[i * args.k + j]} for j in range(args.k)] for i in range(len(problems))]
        summary, recs = summarize_sampled(problems, gens, args.k)
        summary["seconds"] = round(time.time() - t0, 1)
        write_jsonl(os.path.join(out, f"{tag}.jsonl"), recs)
        print(f"[{tag}]", json.dumps(summary, indent=2))
        return summary

    if args.k > 0:
        g = torch.Generator().manual_seed(1234 if args.k_split == "test" else args.seed)
        pool = load_gsm8k(args.k_split)
        perm = torch.randperm(len(pool), generator=g).tolist()     # same subset as train_grpo's pass@k
        subset = [pool[i] for i in (perm[:args.k_n] if args.k_n else range(len(pool)))]
        print(f"[sampled] {len(subset)} {args.k_split} questions x {args.k} samples @ T={args.temperature}")
        results["sampled"] = sampled(subset, "sampled")

    if args.hard and args.k > 0:
        hard = load_hard(args.hard)
        print(f"[hard] {len(hard)} never-solved questions x {args.k} samples")
        results["hard"] = sampled(hard, "hard")
        results["hard"]["solved_now_frac"] = results["hard"][f"pass@{args.k}"]

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {out}/summary.json")


if __name__ == "__main__":
    main()
