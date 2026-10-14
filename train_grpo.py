"""
GRPO on GSM8K with LoRA on Qwen2.5-0.5B-Instruct.

One step:
  1. sample     Q questions x K answers (T=1, no filtering) from the current policy, through an
                engine (--engine hf: Hugging Face generate on the training model; vllm: vLLM with
                the current LoRA adapter, much faster)
  2. score      reward 1 if the boxed answer is right (strict), else 0 (or a coin flip: random)
  3. filter     dynamic sampling: drop all-right / all-wrong groups and refill with new questions
                (at most --refill-rounds extra rounds). All-wrong groups go to hard.jsonl.
  4. weigh      advantage per answer = (r - group mean) / (group std + eps)
  5. update     GRPO loss over micro-batches (gradients add up to the full-batch gradient),
                clip the grad norm, one AdamW step (or --updates-per-batch steps)

Run:
  python train_grpo.py --run baseline                          # defaults = the Phase 2 plan
  python train_grpo.py --run overfit --overfit 4 --steps 40 --lr 1e-4 --eval-every 0
  python train_grpo.py --run baseline --resume                 # continue from the last checkpoint
Output in <out-root>/<run>/: metrics.jsonl, samples.jsonl.gz, hard.jsonl, config.json,
  evals/step<N>.json (per-question correctness), adapters/step<N>/, ckpt/, adapter/ (final)
"""
import argparse
import gzip
import json
import os
import shutil
import time
import uuid

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.data import load_gsm8k
from rl.engine import make_engine
from rl.eval import passk_summary
from rl.grpo import group_advantages, grpo_loss, tis_weights, token_weights
from rl.logprobs import token_logprobs
from rl.reward import reward, score


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output folder name under --out-root")
    ap.add_argument("--out-root", default="runs/grpo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="boxed", choices=["boxed", "hash", "plain"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    # sampling
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--vllm-mem", type=float, default=0.3, help="share of GPU memory vLLM may take")
    ap.add_argument("--vllm-sleep", action="store_true", help="free vLLM's memory while training (small GPUs)")
    ap.add_argument("--questions", type=int, default=16, help="groups trained per step (Q)")
    ap.add_argument("--k", type=int, default=8, help="answers per question (group size K)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gen-batch", type=int, default=256, help="HF engine: max sequences per generate call")
    ap.add_argument("--no-dynamic", action="store_true", help="train on every group, even all-same ones")
    ap.add_argument("--oversample", type=int, default=None,
                    help="extra questions sampled in round 0 (default: half of --questions, i.e. 8 for 16)")
    ap.add_argument("--refill-rounds", type=int, default=2)
    # reward
    ap.add_argument("--reward-mode", default="strict", choices=["strict", "lenient", "random"],
                    help="random = coin flip per answer (control run); accuracy is still the real one")
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
    ap.add_argument("--tis-cap", type=float, default=0.0,
                    help="vLLM only: reweight tokens by p_trainer/p_vllm, capped here (0 = off)")
    # optimization
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=10, help="LR ramps up linearly over this many steps")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--micro-bs", type=int, default=8, help="sequences per forward/backward")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="faster backward, more memory")
    # eval
    ap.add_argument("--eval-every", type=int, default=25, help="0 = never")
    ap.add_argument("--eval-n", type=int, default=0, help="test questions for greedy eval (0 = all 1,319)")
    ap.add_argument("--passk-n", type=int, default=256, help="test questions for pass@k at start and end (0 = off)")
    ap.add_argument("--passk-k", type=int, default=8)
    ap.add_argument("--overfit", type=int, default=0, help="train on this many fixed questions only")
    # bookkeeping
    ap.add_argument("--ckpt-every", type=int, default=25, help="0 = never")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/ckpt")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lm-rl")
    ap.add_argument("--table-per-step", type=int, default=4, help="sample answers per step kept for the wandb table")
    ap.add_argument("--price-per-hour", type=float, default=0.0, help="$ per hour of this machine, for the $ log")
    args = ap.parse_args(argv)
    if args.oversample is None:
        args.oversample = args.questions // 2
    return args


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

    def state_dict(self):
        return {"g": self.g.get_state(), "order": self.order, "pos": self.pos, "seen": self.seen}

    def load_state_dict(self, s):
        self.g.set_state(s["g"])
        self.order, self.pos, self.seen = s["order"], s["pos"], s["seen"]


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
    """Rollout -> per-question list of k sequences {prompt_ids, comp_ids, length, truncated, text
    (, sample_logp)}."""
    P = ro["prompt_len"]
    seqs = []
    for r in range(ro["input_ids"].shape[0]):
        n = int(ro["lengths"][r])
        s = {"prompt_ids": ro["input_ids"][r, :P][ro["attention_mask"][r, :P].bool()],
             "comp_ids": ro["input_ids"][r, P:P + n], "length": n,
             "truncated": bool(ro["truncated"][r]), "text": ro["texts"][r]}
        if "sample_logp" in ro:
            s["sample_logp"] = ro["sample_logp"][r, :n]
        seqs.append(s)
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
    batch = {"input_ids": ids.to(device), "attention_mask": am.to(device), "completion_mask": cm.to(device)}
    extra = None
    if all("sample_logp" in s for s in seqs):
        extra = torch.zeros(len(seqs), T)
        for i, s in enumerate(seqs):
            extra[i, :len(s["sample_logp"])] = s["sample_logp"]
        extra = extra.to(device)
    return batch, P, extra


def sample_groups(engine, stream, problems, args, step, files, reward_gen):
    """Dynamic sampling. Returns the kept groups and round-0 statistics (round 0 is an unbiased
    sample of the current model's accuracy; later rounds are not)."""
    Q, K = args.questions, args.k
    kept, stats = [], {}
    need = Q
    for rnd in range(1 + (0 if args.no_dynamic else args.refill_rounds)):
        n_q = need + (args.oversample if rnd == 0 and not args.no_dynamic else 0)
        idx = stream.next(n_q)
        t0 = time.time()
        ro = engine.sample([problems[i] for i in idx], K, args.temperature)
        for qi, group in zip(idx, split_rows(ro, K)):
            gold = problems[qi]["answer"]
            for s in group:
                s["correct"] = score(s["text"], gold)["correct_strict"]
                if args.reward_mode == "random":
                    s["reward"] = float(torch.rand((), generator=reward_gen) < 0.5)
                else:
                    s["reward"] = reward(s["text"], gold, args.reward_mode, args.format_weight)
                files["samples"].write(json.dumps({"step": step, "round": rnd, "idx": qi, "reward": s["reward"],
                                                   "correct": s["correct"], "length": s["length"],
                                                   "truncated": s["truncated"], "text": s["text"]}) + "\n")
            n_right = sum(s["correct"] for s in group)
            rs = [s["reward"] for s in group]
            if n_right == 0:
                files["hard"].write(json.dumps({"step": step, "idx": qi, "question": problems[qi]["question"],
                                                "answer": gold, "samples": [s["text"] for s in group]}) + "\n")
            if rnd == 0:
                stats.setdefault("rewards", []).extend(rs)
                stats.setdefault("n_right", []).append(n_right)
                stats.setdefault("seqs", []).extend(group)
                stats.setdefault("table", []).extend((qi, s) for s in group[:1])
            useful = args.no_dynamic or max(rs) > min(rs)
            if useful and len(kept) < Q:
                kept.append({"idx": qi, "seqs": group, "rewards": rs})
        need = Q - len(kept)
        stats["rounds"] = rnd + 1
        stats.setdefault("t_rounds", []).append(round(time.time() - t0, 1))
        if need == 0:
            break
    for f in ("hard", "samples"):
        files[f].flush()
    return kept, stats


def train_step(model, opt, sched, groups, args, device, pad_id):
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
        batch, P, _ = collate(mb, pad_id, device)
        f = {}
        with torch.no_grad():
            if args.beta or not args.no_kl_log:              # measure KL even when beta = 0
                with model.disable_adapter():
                    f["ref"] = token_logprobs(model, **batch, prompt_len=P, top_p=None)["logp"]
            if args.updates_per_batch > 1:
                f["old"] = token_logprobs(model, **batch, prompt_len=P, top_p=None)["logp"]
        fixed.append(f)

    keys = ("loss", "pg_part", "kl_part", "kl", "clipped", "entropy", "entropy_top",
            "gap", "gap_abs", "gap_kl", "tis_capped")
    sums = {k: 0.0 for k in keys}
    grad_norms = []
    want_top = args.ent_top_p if (args.ent_coef == 0 or args.ent_kind == "top") else None
    for update in range(args.updates_per_batch):
        opt.zero_grad(set_to_none=True)
        for mb, f in zip(mbs, fixed):
            batch, P, slp = collate(mb, pad_id, device)
            cm = batch["completion_mask"][:, P:]
            out = token_logprobs(model, **batch, prompt_len=P, top_p=want_top,
                                 entropy_grad=args.ent_coef > 0)
            logp = out["logp"]
            old = f.get("old", logp.detach())
            tis = tis_weights(old, slp, args.tis_cap) if (args.tis_cap and slp is not None) else None
            w = token_weights(cm, args.agg, args.max_new_tokens, n_seqs=n_seqs, n_tokens=n_tokens)
            ent = out["entropy_top" if args.ent_kind == "top" else "entropy"]
            loss, info = grpo_loss(logp, old, f.get("ref"), torch.tensor([s["adv"] for s in mb], device=device),
                                   w, args.clip_low, args.clip_high, args.beta, ent, args.ent_coef, tis)
            loss.backward()
            if update == 0:                                             # log the first pass only
                m = cm.float()
                sums["loss"] += loss.item()
                sums["pg_part"] += info["pg_part"]
                sums["kl_part"] += info["kl_part"]
                for key, val in (("kl", info["kl"]), ("clipped", info["clipped"]),
                                 ("entropy", out["entropy"]), ("entropy_top", out["entropy_top"])):
                    sums[key] += (val.detach() * m).sum().item()
                if slp is not None:                                     # trainer vs vLLM, same tokens
                    d = (logp.detach() - slp) * m
                    sums["gap"] += d.sum().item()
                    sums["gap_abs"] += d.abs().sum().item()
                    sums["gap_kl"] += ((torch.exp(-d) + d - 1) * m).sum().item()   # k3, vLLM vs trainer
                    if tis is not None:
                        sums["tis_capped"] += ((tis >= args.tis_cap) * m).sum().item()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        grad_norms.append(gn.item())
        opt.step()
        sched.step()
    stats = {"loss": sums["loss"], "pg_part": sums["pg_part"], "kl_part": sums["kl_part"],
             "grad_norm": grad_norms[0], "lr": sched.get_last_lr()[0],
             **{k: sums[k] / n_tokens for k in ("kl", "clipped", "entropy", "entropy_top")}}
    stats["clip_frac"] = stats.pop("clipped")
    if any("sample_logp" in s for s in seqs):
        stats.update({k: sums[k] / n_tokens for k in ("gap", "gap_abs", "gap_kl")})
        if args.tis_cap:
            stats["tis_capped_frac"] = sums["tis_capped"] / n_tokens
    return stats


@torch.no_grad()
def greedy_eval(engine, model, problems):
    """-> (summary dict, per-question 0/1 list in problem order, for paired comparisons)."""
    model.eval()
    gens = engine.greedy(problems)
    correct = [int(score(g["text"], p["answer"])["correct_strict"]) for g, p in zip(gens, problems)]
    return {"eval_acc": sum(correct) / len(correct),
            "eval_len": sum(g["length"] for g in gens) / len(gens),
            "eval_trunc": sum(g["truncated"] for g in gens) / len(gens)}, correct


@torch.no_grad()
def passk_eval(engine, model, problems, k, temperature):
    model.eval()
    ro = engine.sample(problems, k, temperature)
    counts = [sum(score(s["text"], p["answer"])["correct_strict"] for s in group)
              for p, group in zip(problems, split_rows(ro, k))]
    return passk_summary(counts, k)


def save_checkpoint(path, model, opt, sched, stream, reward_gen, step, extra):
    """Write to <path>.tmp, then swap it in, so a crash mid-write never leaves a broken checkpoint."""
    tmp, old = path + ".tmp", path + ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "stream": stream.state_dict(),
                "reward_gen": reward_gen.get_state(), "step": step,
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(), **extra},
               os.path.join(tmp, "state.pt"))
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(path):
        os.rename(path, old)
    os.rename(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def load_checkpoint(path, model, opt, sched, stream, reward_gen):
    if not os.path.exists(path) and os.path.exists(path + ".old"):
        path = path + ".old"
    set_peft_model_state_dict(model, load_file(os.path.join(path, "adapter_model.safetensors")))
    st = torch.load(os.path.join(path, "state.pt"), weights_only=False)
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    stream.load_state_dict(st["stream"])
    reward_gen.set_state(st["reward_gen"])
    torch.set_rng_state(st["rng_cpu"])
    torch.cuda.set_rng_state_all(st["rng_cuda"])
    return st


def trim_logs(out, step):
    """On resume, drop log lines written after the checkpoint (they will be redone)."""
    for name, opener in (("metrics.jsonl", open), ("hard.jsonl", open), ("samples.jsonl.gz", gzip.open)):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            continue
        with opener(p, "rt") as f:
            keep = [l for l in f if json.loads(l)["step"] <= step]
        with opener(p, "wt") as f:
            f.writelines(keep)


def main(argv=None, on_checkpoint=None):
    """on_checkpoint: optional callback after every checkpoint (Modal uses it to save its disk)."""
    args = parse_args(argv)
    out = os.path.join(args.out_root, args.run)
    ckpt_dir = os.path.join(out, "ckpt")
    if os.path.exists(os.path.join(out, "metrics.jsonl")) and not args.resume:
        raise SystemExit(f"{out} already has a run. Use --resume to continue it, or a new --run name.")
    os.makedirs(out, exist_ok=True)
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
    # Constant LR after a short linear warmup: step 1 uses lr/warmup, step `warmup` the full lr.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) if args.warmup else 1.0)

    train = load_gsm8k("train")
    if args.overfit:
        keep = pick_overfit_questions(train, args.overfit)
        train = [train[i] for i in keep]
        args.no_dynamic, args.questions = True, args.overfit
    stream = QuestionStream(train, args.seed)
    reward_gen = torch.Generator().manual_seed(args.seed + 12345)
    test = load_gsm8k("test")
    g = torch.Generator().manual_seed(1234)                       # fixed subsets for every run
    perm = torch.randperm(len(test), generator=g).tolist()
    eval_set = [test[i] for i in (perm[:args.eval_n] if args.eval_n else range(len(test)))]
    passk_set = [test[i] for i in perm[:args.passk_n]]

    start, prev_elapsed, wandb_id = 0, 0.0, None
    if args.resume and (os.path.exists(ckpt_dir) or os.path.exists(ckpt_dir + ".old")):
        st = load_checkpoint(ckpt_dir, model, opt, sched, stream, reward_gen)
        start, prev_elapsed, wandb_id = st["step"], st["elapsed"], st.get("wandb_id")
        trim_logs(out, start)
        print(f"resumed from step {start}")
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)

    engine = make_engine(args.engine, model, tok, args, out)       # vLLM gets the (resumed) adapter
    wb = None
    if args.wandb:
        import wandb
        wandb_id = wandb_id or uuid.uuid4().hex[:8]
        wb = wandb.init(project=args.wandb_project, name=args.run, id=wandb_id, resume="allow", config=vars(args))
        table_rows = []

    files = {"hard": open(os.path.join(out, "hard.jsonl"), "a"),
             "samples": gzip.open(os.path.join(out, "samples.jsonl.gz"), "at")}
    log = open(os.path.join(out, "metrics.jsonl"), "a")
    t_start = time.time()

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if wb:
            wb.log({k: v for k, v in rec.items() if isinstance(v, (int, float)) and k not in ("step", "type")},
                   step=rec["step"])

    def evaluate(step):
        t0 = time.time()
        ev, correct = greedy_eval(engine, model, eval_set)
        os.makedirs(os.path.join(out, "evals"), exist_ok=True)
        json.dump({"step": step, "n": len(correct), "correct": correct}, open(os.path.join(out, "evals", f"step{step}.json"), "w"))
        write({"type": "eval", "step": step, **ev, "t": time.time() - t0})
        print(f"[eval] step {step} acc={ev['eval_acc']:.3f} len={ev['eval_len']:.0f} ({time.time() - t0:.0f}s)")

    def passk(step, tag):
        t0 = time.time()
        pk = passk_eval(engine, model, passk_set, args.passk_k, args.temperature)
        write({"type": "passk", "step": step, "when": tag, **{f"test_{k}": v for k, v in pk.items()},
               "t": time.time() - t0})
        print(f"[pass@k] {tag}: " + " ".join(f"{k}={v:.3f}" for k, v in pk.items()))

    if start == 0:
        if args.eval_every:
            evaluate(0)
        if args.passk_n:
            passk(0, "start")

    for step in range(start + 1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.eval()
        groups, st = sample_groups(engine, stream, train, args, step, files, reward_gen)
        engine.release()
        t_gen = time.time() - t0
        rec = {"type": "train", "step": step}
        nr, K = st["n_right"], args.k
        s0 = st["seqs"]
        fin = [s["length"] for s in s0 if not s["truncated"]]
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
            "len_finished": sum(fin) / max(1, len(fin)),
            "len_min": min(s["length"] for s in s0), "len_max": max(s["length"] for s in s0),
            "trunc_frac": sum(s["truncated"] for s in s0) / len(s0),
            "n_right": nr,
            "questions_seen": stream.seen,
        })
        if groups:
            model.train()
            t1 = time.time()
            rec.update(train_step(model, opt, sched, groups, args, device, tok.pad_token_id))
            rec["t_train"] = time.time() - t1
        t2 = time.time()
        engine.sync(step)                                           # next samples come from the new policy
        rec["t_sync"] = time.time() - t2
        elapsed = prev_elapsed + time.time() - t_start
        rec.update({"t_gen": t_gen, "t_step": time.time() - t0, "hours": elapsed / 3600,
                    "dollars": elapsed / 3600 * args.price_per_hour,
                    "vram_gb": torch.cuda.max_memory_allocated() / 2**30})
        write(rec)
        if wb:
            table_rows += [[step, qi, s["reward"], int(s["correct"]), s["length"], s["text"]]
                           for qi, s in st["table"][:args.table_per_step]]
        print(f"step {step:4d} | acc {rec['acc']:.3f} | wasted {rec['wasted_frac']:.2f} r{rec['rounds']} "
              f"| len {rec['len']:.0f} | kl {rec.get('kl', 0):.4f} | H {rec.get('entropy', 0):.3f}/"
              f"{rec.get('entropy_top', 0):.3f} | gn {rec.get('grad_norm', 0):.3f} "
              f"| {rec['t_gen']:.0f}s+{rec.get('t_train', 0):.0f}s | {rec['vram_gb']:.1f}GB", flush=True)
        for v in ("loss", "kl", "grad_norm"):
            assert v not in rec or rec[v] == rec[v], f"NaN in {v} at step {step}"

        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            evaluate(step)
            model.save_pretrained(os.path.join(out, "adapters", f"step{step}"))
            if wb:
                import wandb
                wb.log({"samples": wandb.Table(columns=["step", "question", "reward", "correct", "length", "answer"],
                                               data=table_rows)}, step=step)
        if args.ckpt_every and (step % args.ckpt_every == 0 or step == args.steps):
            save_checkpoint(ckpt_dir, model, opt, sched, stream, reward_gen, step,
                            {"elapsed": prev_elapsed + time.time() - t_start, "wandb_id": wandb_id})
            if on_checkpoint:
                on_checkpoint()

    end_logged = any('"when": "end"' in l for l in open(os.path.join(out, "metrics.jsonl")))
    if args.passk_n and not end_logged:                  # also after a crash between the last checkpoint and here
        passk(args.steps, "end")
    model.save_pretrained(os.path.join(out, "adapter"))
    for f in (*files.values(), log):
        f.close()
    if wb:
        wb.finish()
    if on_checkpoint:
        on_checkpoint()


if __name__ == "__main__":
    main()
