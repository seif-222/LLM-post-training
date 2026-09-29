"""
GRPO on GSM8K with LoRA on Qwen2.5-0.5B-Instruct.

One step:
  1. sample     Q questions x K answers (T=1, no filtering), with LoRA on, no gradient
  2. score      reward 1 if the boxed answer is right (strict), else 0
  3. filter     dynamic sampling: drop all-right / all-wrong groups and refill with new questions
                (at most --refill-rounds extra rounds). All-wrong groups go to hard.jsonl.
  4. weigh      advantage per answer = (r - group mean) / (group std + eps)
  5. update     GRPO loss over micro-batches (gradients add up to the full-batch gradient),
                clip the grad norm, one AdamW step (or --updates-per-batch steps)

Run:
  python train_grpo.py --run baseline                          # defaults = the Phase 2 plan
  python train_grpo.py --run overfit --overfit 4 --steps 40 --lr 1e-4 --eval-every 0
Output in runs/grpo/<run>/: metrics.jsonl, hard.jsonl, config.json, adapter/
"""
import argparse
import json
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.data import build_prompt, load_gsm8k
from rl.eval import gen_config, generate
from rl.grpo import group_advantages, grpo_loss, token_weights
from rl.logprobs import token_logprobs
from rl.reward import reward, score
from rl.rollout import rollout


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output folder name under runs/grpo/")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="boxed", choices=["boxed", "hash"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    # sampling
    ap.add_argument("--questions", type=int, default=16, help="groups trained per step (Q)")
    ap.add_argument("--k", type=int, default=8, help="answers per question (group size K)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gen-batch", type=int, default=256, help="max sequences per generate call")
    ap.add_argument("--no-dynamic", action="store_true", help="train on every group, even all-same ones")
    ap.add_argument("--oversample", type=int, default=8, help="extra questions sampled in round 0")
    ap.add_argument("--refill-rounds", type=int, default=2)
    # reward
    ap.add_argument("--reward-mode", default="strict", choices=["strict", "lenient"])
    ap.add_argument("--format-weight", type=float, default=0.0)
    # loss
    ap.add_argument("--no-std", action="store_true", help="advantage without dividing by the group std")
    ap.add_argument("--agg", default="seq", choices=["seq", "token", "const"])
    ap.add_argument("--beta", type=float, default=0.02, help="KL leash strength")
    ap.add_argument("--no-kl-log", action="store_true",
                    help="with beta=0, skip the reference pass (faster, but KL is then not measured)")
    ap.add_argument("--clip-low", type=float, default=0.2)
    ap.add_argument("--clip-high", type=float, default=0.2)
    ap.add_argument("--updates-per-batch", type=int, default=1, help=">1 reuses samples; then clipping matters")
    ap.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus (off by default)")
    ap.add_argument("--ent-kind", default="top", choices=["full", "top"])
    ap.add_argument("--ent-top-p", type=float, default=0.9)
    # optimization
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--micro-bs", type=int, default=8, help="sequences per forward/backward")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="faster backward, more memory")
    # eval
    ap.add_argument("--eval-every", type=int, default=25, help="0 = never")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--overfit", type=int, default=0, help="train on this many fixed questions only")
    return ap.parse_args(argv)


class QuestionStream:
    """Train questions in a shuffled order, reshuffled every epoch."""

    def __init__(self, problems, seed):
        self.problems, self.g = problems, torch.Generator().manual_seed(seed)
        self.order, self.pos, self.seen = [], 0, 0

    def next(self, n):
        out = []
        while len(out) < n:
            if self.pos == len(self.order):
                self.order, self.pos = torch.randperm(len(self.problems), generator=self.g).tolist(), 0
            out.append(self.order[self.pos])
            self.pos += 1
        self.seen += n
        return out


def pick_overfit_questions(problems, n):
    """Questions the base model got partly right in Phase 1 (2-5 of 8), so there is room to learn."""
    path = "runs/eval/Qwen2.5-0.5B-Instruct_boxed/sampled.jsonl"
    by_q = {p["question"]: i for i, p in enumerate(problems)}
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        ids = [by_q[r["question"]] for r in rows if 2 <= r["n_correct"] <= 5 and r["question"] in by_q]
        if len(ids) >= n:
            return ids[:n]
    return list(range(n))


def split_rows(ro, k):
    """Rollout -> per-question list of k sequences {prompt_ids, comp_ids, length, truncated, text}."""
    P = ro["prompt_len"]
    seqs = []
    for r in range(ro["input_ids"].shape[0]):
        n = int(ro["lengths"][r])
        seqs.append({"prompt_ids": ro["input_ids"][r, :P][ro["attention_mask"][r, :P].bool()],
                     "comp_ids": ro["input_ids"][r, P:P + n], "length": n,
                     "truncated": bool(ro["truncated"][r]), "text": ro["texts"][r]})
    return [seqs[i:i + k] for i in range(0, len(seqs), k)]


def collate(seqs, pad_id, device):
    """Left-pad prompts, right-pad answers -> the same layout rollout() produces."""
    P = max(len(s["prompt_ids"]) for s in seqs)
    T = max(len(s["comp_ids"]) for s in seqs)
    ids = torch.full((len(seqs), P + T), pad_id, dtype=torch.long)
    am = torch.zeros_like(ids)
    cm = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        p, c = len(s["prompt_ids"]), len(s["comp_ids"])
        ids[i, P - p:P], ids[i, P:P + c] = s["prompt_ids"], s["comp_ids"]
        am[i, P - p:P + c] = 1
        cm[i, P:P + c] = 1
    return {"input_ids": ids.to(device), "attention_mask": am.to(device),
            "completion_mask": cm.to(device)}, P


def sample_groups(model, tok, stream, problems, args, cfg, step, hard_f):
    """Dynamic sampling. Returns the kept groups and round-0 statistics (round 0 is an unbiased
    sample of the current model's accuracy; later rounds are not)."""
    Q, K = args.questions, args.k
    kept, stats = [], {}
    need = Q
    for rnd in range(1 + (0 if args.no_dynamic else args.refill_rounds)):
        n_q = need + (args.oversample if rnd == 0 and not args.no_dynamic else 0)
        idx = stream.next(n_q)
        t0 = time.time()
        ro = rollout(model, tok, [problems[i] for i in idx], K, cfg, args.prompt, args.gen_batch)
        for qi, group in zip(idx, split_rows(ro, K)):
            gold = problems[qi]["answer"]
            for s in group:
                s["reward"] = reward(s["text"], gold, args.reward_mode, args.format_weight)
                s["correct"] = score(s["text"], gold)["correct_strict"]
            n_right = sum(s["correct"] for s in group)
            rs = [s["reward"] for s in group]
            if n_right == 0:
                hard_f.write(json.dumps({"step": step, "idx": qi, "question": problems[qi]["question"],
                                         "answer": gold, "samples": [s["text"] for s in group]}) + "\n")
            if rnd == 0:
                stats.setdefault("rewards", []).extend(rs)
                stats.setdefault("n_right", []).append(n_right)
                stats.setdefault("seqs", []).extend(group)
            useful = args.no_dynamic or max(rs) > min(rs)
            if useful and len(kept) < Q:
                kept.append({"idx": qi, "seqs": group, "rewards": rs})
        need = Q - len(kept)
        stats["rounds"] = rnd + 1
        stats.setdefault("t_rounds", []).append(round(time.time() - t0, 1))
        if need == 0:
            break
    hard_f.flush()
    return kept, stats


def train_step(model, opt, groups, args, device, pad_id):
    """Advantages -> token weights over the whole batch -> micro-batched loss/backward."""
    K = args.k
    seqs = [s for g in groups for s in g["seqs"]]
    rewards = torch.tensor([s["reward"] for s in seqs])
    adv = group_advantages(rewards, K, scale_std=not args.no_std)
    for s, a in zip(seqs, adv):
        s["adv"] = float(a)
    n_seqs, n_tokens = len(seqs), sum(s["length"] for s in seqs)
    order = sorted(range(n_seqs), key=lambda i: seqs[i]["length"])     # similar lengths share padding
    mbs = [[seqs[i] for i in order[j:j + args.micro_bs]] for j in range(0, n_seqs, args.micro_bs)]

    # Fixed per-batch quantities: reference log-probs (LoRA off) and, if samples are reused,
    # the log-probs at sampling time. Both without gradient.
    fixed = []
    for mb in mbs:
        batch, P = collate(mb, pad_id, device)
        f = {}
        with torch.no_grad():
            if args.beta or not args.no_kl_log:              # measure KL even when beta = 0
                with model.disable_adapter():
                    f["ref"] = token_logprobs(model, **batch, prompt_len=P, top_p=None)["logp"]
            if args.updates_per_batch > 1:
                f["old"] = token_logprobs(model, **batch, prompt_len=P, top_p=None)["logp"]
        fixed.append(f)

    sums = {k: 0.0 for k in ("loss", "kl", "clipped", "entropy", "entropy_top")}
    grad_norms = []
    want_top = args.ent_top_p if (args.ent_coef == 0 or args.ent_kind == "top") else None
    for update in range(args.updates_per_batch):
        opt.zero_grad(set_to_none=True)
        for mb, f in zip(mbs, fixed):
            batch, P = collate(mb, pad_id, device)
            cm = batch["completion_mask"][:, P:]
            out = token_logprobs(model, **batch, prompt_len=P, top_p=want_top,
                                 entropy_grad=args.ent_coef > 0)
            logp = out["logp"]
            old = f.get("old", logp.detach())
            w = token_weights(cm, args.agg, args.max_new_tokens, n_seqs=n_seqs, n_tokens=n_tokens)
            ent = out["entropy_top" if args.ent_kind == "top" else "entropy"]
            loss, info = grpo_loss(logp, old, f.get("ref"), torch.tensor([s["adv"] for s in mb], device=device),
                                   w, args.clip_low, args.clip_high, args.beta, ent, args.ent_coef)
            loss.backward()
            if update == 0:                                             # log the first pass only
                m = cm.float()
                sums["loss"] += loss.item()
                for key, val in (("kl", info["kl"]), ("clipped", info["clipped"]),
                                 ("entropy", out["entropy"]), ("entropy_top", out["entropy_top"])):
                    sums[key] += (val.detach() * m).sum().item()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        grad_norms.append(gn.item())
        opt.step()
    stats = {"loss": sums["loss"], "grad_norm": grad_norms[0],
             **{k: sums[k] / n_tokens for k in ("kl", "clipped", "entropy", "entropy_top")}}
    stats["clip_frac"] = stats.pop("clipped")
    return stats


@torch.no_grad()
def greedy_eval(model, tok, problems, args):
    model.eval()
    prompts = [build_prompt(tok, p["question"], args.prompt) for p in problems]
    gens = generate(model, tok, prompts, gen_config(model, False, None, args.max_new_tokens),
                    batch_size=200, verbose=False)
    correct = [score(g[0]["text"], p["answer"])["correct_strict"] for g, p in zip(gens, problems)]
    return {"eval_acc": sum(correct) / len(correct),
            "eval_len": sum(g[0]["length"] for g in gens) / len(gens),
            "eval_trunc": sum(g[0]["truncated"] for g in gens) / len(gens)}


def main(argv=None):
    args = parse_args(argv)
    out = os.path.join("runs", "grpo", args.run)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)
    torch.manual_seed(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                             target_modules="all-linear", lora_dropout=0.0))
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = True                  # generate() needs the KV cache; training passes use_cache=False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA params: {n_train / 1e6:.2f}M")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            betas=(0.9, 0.99), weight_decay=0.0)
    cfg = gen_config(model, True, args.temperature, args.max_new_tokens)

    train = load_gsm8k("train")
    if args.overfit:
        keep = pick_overfit_questions(train, args.overfit)
        train = [train[i] for i in keep]
        args.no_dynamic, args.questions = True, args.overfit
    stream = QuestionStream(train, args.seed)
    test = load_gsm8k("test")
    g = torch.Generator().manual_seed(1234)                       # fixed eval subset for every run
    eval_set = [test[i] for i in torch.randperm(len(test), generator=g)[:args.eval_n].tolist()]

    log = open(os.path.join(out, "metrics.jsonl"), "a")
    hard_f = open(os.path.join(out, "hard.jsonl"), "a")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    if args.eval_every:
        t0 = time.time()
        ev = greedy_eval(model, tok, eval_set, args)
        write({"type": "eval", "step": 0, **ev, "t": time.time() - t0})
        print(f"[eval] step 0 acc={ev['eval_acc']:.3f} len={ev['eval_len']:.0f}")

    for step in range(1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.eval()
        groups, st = sample_groups(model, tok, stream, train, args, cfg, step, hard_f)
        t_gen = time.time() - t0
        rec = {"type": "train", "step": step}
        nr, K = st["n_right"], args.k
        s0 = st["seqs"]
        rec.update({
            "reward": sum(st["rewards"]) / len(st["rewards"]),
            "acc": sum(s["correct"] for s in s0) / len(s0),
            "wasted_frac": sum(c in (0, K) for c in nr) / len(nr),
            "all_right_frac": sum(c == K for c in nr) / len(nr),
            "all_wrong_frac": sum(c == 0 for c in nr) / len(nr),
            "rounds": st["rounds"], "t_rounds": st["t_rounds"], "n_groups": len(groups),
            "len": sum(s["length"] for s in s0) / len(s0),
            "len_right": sum(s["length"] for s in s0 if s["correct"]) / max(1, sum(s["correct"] for s in s0)),
            "len_wrong": sum(s["length"] for s in s0 if not s["correct"]) / max(1, sum(not s["correct"] for s in s0)),
            "trunc_frac": sum(s["truncated"] for s in s0) / len(s0),
            "n_right": nr,
            "questions_seen": stream.seen,
        })
        if groups:
            model.train()
            t1 = time.time()
            rec.update(train_step(model, opt, groups, args, device, tok.pad_token_id))
            rec["t_train"] = time.time() - t1
        rec.update({"t_gen": t_gen, "t_step": time.time() - t0,
                    "vram_gb": torch.cuda.max_memory_allocated() / 2**30})
        write(rec)
        print(f"step {step:4d} | acc {rec['acc']:.3f} | wasted {rec['wasted_frac']:.2f} r{rec['rounds']} "
              f"| len {rec['len']:.0f} | kl {rec.get('kl', 0):.4f} | H {rec.get('entropy', 0):.3f}/"
              f"{rec.get('entropy_top', 0):.3f} | gn {rec.get('grad_norm', 0):.3f} "
              f"| {rec['t_gen']:.0f}s+{rec.get('t_train', 0):.0f}s | {rec['vram_gb']:.1f}GB", flush=True)
        for v in ("loss", "kl", "grad_norm"):
            assert v not in rec or rec[v] == rec[v], f"NaN in {v} at step {step}"

        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            t0 = time.time()
            ev = greedy_eval(model, tok, eval_set, args)
            write({"type": "eval", "step": step, **ev, "t": time.time() - t0})
            print(f"[eval] step {step} acc={ev['eval_acc']:.3f} len={ev['eval_len']:.0f}")
            model.save_pretrained(os.path.join(out, "adapter"))
    model.save_pretrained(os.path.join(out, "adapter"))


if __name__ == "__main__":
    main()
