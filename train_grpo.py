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
from rl.grpo import group_advantages, grpo_loss, tis_weights, token_weights, mask_advantages, gradient_diagnostics, kl_k3
from rl.campaign import isolated_rng, training_requests, evaluation_requests, finite, config_hash, assert_model_setup
from rl import checkpointing
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
    ap.add_argument("--advantage-mode", choices=["full", "positive", "negative"], default="full")
    ap.add_argument("--campaign-id", default=None)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--data-revision", default=None)
    ap.add_argument("--diagnostic-updates", default="25,100,200")
    ap.add_argument("--max-attempts", type=int, default=220)
    ap.add_argument("--max-empty-attempts", type=int, default=3)
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
    kept, stats = [], {"generated_tokens": 0, "generated_answers": 0}
    need = Q
    for rnd in range(1 + (0 if args.no_dynamic else args.refill_rounds)):
        n_q = need + (args.oversample if rnd == 0 and not args.no_dynamic else 0)
        idx = stream.next(n_q)
        t0 = time.time()
        requests = training_requests(args.seed, step, rnd, [problems[i].get("id", i) for i in idx])
        ro = engine.sample([problems[i] for i in idx], K, args.temperature, requests=requests)
        stats["generated_tokens"] += int(ro["lengths"].sum())
        stats["generated_answers"] += len(ro["lengths"])
        for qi, group, req in zip(idx, split_rows(ro, K), requests):
            gold = problems[qi]["answer"]
            for s in group:
                scoring = score(s["text"], gold)
                s["correct"] = scoring["correct_strict"]
                s["formatted"] = scoring["source"] in ("boxed", "hash")
                if args.reward_mode == "random":
                    s["reward"] = float(torch.rand((), generator=reward_gen) < 0.5)
                else:
                    s["reward"] = reward(s["text"], gold, args.reward_mode, args.format_weight)
                files["samples"].write(json.dumps({"step": getattr(args, "current_update", step), "rollout_attempt": step, "round": rnd, "idx": qi, "reward": s["reward"],
                                                   "correct": s["correct"], "length": s["length"],
                                                   "truncated": s["truncated"], "text": s["text"], "request": req,
                                                   "token_ids": s["comp_ids"].tolist()}) + "\n")
            n_right = sum(s["correct"] for s in group)
            rs = [s["reward"] for s in group]
            if n_right == 0:
                files["hard"].write(json.dumps({"step": getattr(args, "current_update", step), "rollout_attempt": step, "idx": qi, "question": problems[qi]["question"],
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
    finite(rewards, "rewards")
    raw_adv = group_advantages(rewards, K, scale_std=not args.no_std)
    adv = mask_advantages(raw_adv, args.advantage_mode)
    finite(adv, "advantages")
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
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    lr_used = opt.param_groups[0]["lr"]
    diagnostic = getattr(args, "current_update", -1) in {int(x) for x in args.diagnostic_updates.split(",") if x}
    kl_grad = [torch.zeros_like(p) for p in params] if diagnostic else None
    diagnostics = {}
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
            finite([logp, old, f.get("ref"), loss], "loss inputs")
            if diagnostic:
                with isolated_rng():
                    kl_loss = args.beta * (w * kl_k3(logp, f["ref"])).sum() if args.beta else logp.sum() * 0
                    grads = torch.autograd.grad(kl_loss, params, retain_graph=True, allow_unused=True)
                    for target, grad in zip(kl_grad, grads):
                        if grad is not None:
                            target.add_(grad.detach())
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
        finite([p.grad for p in params], "gradient")
        if diagnostic:
            diagnostics = gradient_diagnostics([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params], kl_grad)
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip, error_if_nonfinite=True)
        grad_norms.append(gn.item())
        opt.step()
        finite([p.detach() for p in params], "updated parameters")
        finite(opt.state_dict(), "optimizer")
        sched.step()
    stats = {"loss": sums["loss"], "pg_part": sums["pg_part"], "kl_part": sums["kl_part"],
             "grad_norm": grad_norms[0], "lr": lr_used,
             "grad_clip_active": grad_norms[0] > args.grad_clip,
             "objective_kl": sums["kl_part"] / args.beta if args.beta else None,
             "parameter_rms_change": (sum((p.detach() - b).double().square().sum().item() for p, b in zip(params, before)) / sum(p.numel() for p in params)) ** 0.5,
             "active_positive_answers": int((adv > 0).sum()), "active_negative_answers": int((adv < 0).sum()),
             "active_positive_tokens": sum(s["length"] for s in seqs if s["adv"] > 0),
             "active_negative_tokens": sum(s["length"] for s in seqs if s["adv"] < 0),
             "trained_tokens": n_tokens, "trained_answers": n_seqs,
             "adv_mean_before": raw_adv.mean().item(), "adv_std_before": raw_adv.std(correction=0).item(),
             "adv_mean_after": adv.mean().item(), "adv_std_after": adv.std(correction=0).item(),
             **diagnostics,
             **{k: sums[k] / n_tokens for k in ("kl", "clipped", "entropy", "entropy_top")}}
    stats["clip_frac"] = stats.pop("clipped")
    if any("sample_logp" in s for s in seqs):
        stats.update({k: sums[k] / n_tokens for k in ("gap", "gap_abs", "gap_kl")})
        if args.tis_cap:
            stats["tis_capped_frac"] = sums["tis_capped"] / n_tokens
    return stats


@torch.no_grad()
def greedy_eval(engine, model, problems, return_records=False):
    """-> (summary dict, per-question 0/1 list in problem order, for paired comparisons)."""
    model.eval()
    gens = engine.greedy(problems)
    correct = [int(score(g["text"], p["answer"])["correct_strict"]) for g, p in zip(gens, problems)]
    summary = {"eval_acc": sum(correct) / len(correct),
               "eval_len": sum(g["length"] for g in gens) / len(gens),
               "eval_trunc": sum(g["truncated"] for g in gens) / len(gens)}
    if return_records:
        requests = evaluation_requests("openai/gsm8k/main/test", [p.get("id", i) for i,p in enumerate(problems)])
        records = [{"id": p.get("id", i), "question": p["question"], "answer": p["answer"],
                    **g, **score(g["text"], p["answer"]), "request": req}
                   for i,(p,g,req) in enumerate(zip(problems,gens,requests))]
        return summary, correct, records
    return summary, correct



@torch.no_grad()
def passk_eval(engine, model, problems, k, temperature):
    model.eval()
    ro = engine.sample(problems, k, temperature)
    counts = [sum(score(s["text"], p["answer"])["correct_strict"] for s in group)
              for p, group in zip(problems, split_rows(ro, k))]
    return passk_summary(counts, k)


def save_checkpoint(path, model, opt, sched, stream, reward_gen, step, extra, commit=None):
    return checkpointing.save(path, model, opt, sched, stream, reward_gen, step, extra, commit)


def load_checkpoint(path, model, opt, sched, stream, reward_gen, expected_config=None):
    return checkpointing.load(path, model, opt, sched, stream, reward_gen, expected_config)


def trim_logs(out, step):
    """Preserve raw files, then rebuild the legacy canonical view through the checkpoint."""
    for name, opener in (("metrics.jsonl", open), ("hard.jsonl", open), ("samples.jsonl.gz", gzip.open)):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            continue
        shutil.copy2(p, p + ".attempt-" + uuid.uuid4().hex)
        with opener(p, "rt") as f:
            keep = [l for l in f if json.loads(l)["step"] <= step]
        with opener(p, "wt") as f:
            f.writelines(keep)


def main(argv=None, on_checkpoint=None, guard=None, _test_device=None):
    """on_checkpoint: optional callback after every checkpoint (Modal uses it to save its disk)."""
    startup = time.monotonic()
    args = parse_args(argv)
    expected_config = config_hash(args)
    from pathlib import Path
    source_root = Path(__file__).resolve().parent
    source_fingerprints = {str(path.relative_to(source_root)): checkpointing.file_hash(path)
                           for path in [source_root/"train_grpo.py", *sorted((source_root/"rl").glob("*.py"))]}
    out = os.path.join(args.out_root, args.run)
    ckpt_dir = os.path.join(out, "ckpt")
    if os.path.exists(os.path.join(out, "metrics.jsonl")) and not args.resume:
        raise SystemExit(f"{out} already has a run. Use --resume to continue it, or a new --run name.")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = _test_device or "cuda"

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                             target_modules="all-linear", lora_dropout=0.0))
    if args.campaign_id:
        checkpointing.atomic_json(os.path.join(out, "model-setup.json"), assert_model_setup(model))
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = True                  # generate() needs the KV cache; training passes use_cache=False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA params: {n_train / 1e6:.2f}M")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
    # Constant LR after a short linear warmup: step 1 uses lr/warmup, step `warmup` the full lr.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) if args.warmup else 1.0)

    train = load_gsm8k("train", revision=args.data_revision)
    if args.overfit:
        keep = pick_overfit_questions(train, args.overfit)
        train = [train[i] for i in keep]
        args.no_dynamic, args.questions = True, args.overfit
    stream = QuestionStream(train, args.seed)
    reward_gen = torch.Generator().manual_seed(args.seed + 12345)
    test = load_gsm8k("test", revision=args.data_revision)
    g = torch.Generator().manual_seed(1234)                       # fixed subsets for every run
    perm = torch.randperm(len(test), generator=g).tolist()
    eval_set = [test[i] for i in (perm[:args.eval_n] if args.eval_n else range(len(test)))]
    passk_set = [test[i] for i in perm[:args.passk_n]]

    lineage = {}
    initial_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    start, attempt, empty_attempts, gap_strikes, prev_elapsed, wandb_id = 0, 0, 0, 0, 0.0, None
    if args.resume:
        st = load_checkpoint(ckpt_dir, model, opt, sched, stream, reward_gen, expected_config)
        start, prev_elapsed, wandb_id = st["step"], st["elapsed"], st.get("wandb_id")
        if st.get("source_hashes") != source_fingerprints:
            raise ValueError("checkpoint source hash mismatch")
        attempt = st.get("attempt", start)
        empty_attempts, gap_strikes = st.get("empty_attempts", 0), st.get("gap_strikes", 0)
        initial_params = [p.to(device) for p in st.get("initial_params", initial_params)]
        lineage = st.get("log_lineage", {})
        if lineage:
            checkpointing.restore_logs(out, lineage)
        else:
            trim_logs(out, start)
        print(f"resumed from step {start}")
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)

    with isolated_rng():
        engine = make_engine(args.engine, model, tok, args, out)       # vLLM gets the (resumed) adapter
    wb = None
    if args.wandb:
        import wandb
        wandb_id = uuid.uuid4().hex[:8]
        wb = wandb.init(project=args.wandb_project, name=args.run + "/" + wandb_id, id=wandb_id,
                        group=(args.campaign_id or "legacy") + "/" + args.run, config=vars(args),
                        settings=wandb.Settings(init_timeout=60))
        table_rows = []

    attempt_id = wandb_id or uuid.uuid4().hex[:8]
    raw = os.path.join(out, "attempts", attempt_id)
    os.makedirs(raw, exist_ok=False)
    raw_metrics = checkpointing.ChunkLog(os.path.join(raw, "metrics"))
    raw_samples = checkpointing.ChunkLog(os.path.join(raw, "samples"))
    raw_hard = checkpointing.ChunkLog(os.path.join(raw, "hard"))
    files = {"hard": checkpointing.MirrorLog(open(os.path.join(out, "hard.jsonl"), "a"), raw_hard),
             "samples": checkpointing.MirrorLog(gzip.open(os.path.join(out, "samples.jsonl.gz"), "at"), raw_samples)}
    log = open(os.path.join(out, "metrics.jsonl"), "a")
    t_start = time.time()
    telemetry_failed_at = None
    last_telemetry_check = 0.0

    def checkpoint(step):
        save_started = time.monotonic()
        for f in (*files.values(), log, raw_metrics):
            f.flush()
        lineage[attempt_id] = {"metrics": raw_metrics.index, "samples": raw_samples.index, "hard": raw_hard.index}
        save_checkpoint(ckpt_dir, model, opt, sched, stream, reward_gen, step,
                        {"elapsed": prev_elapsed + time.time() - t_start, "wandb_id": wandb_id,
                         "attempt": attempt, "empty_attempts": empty_attempts, "gap_strikes": gap_strikes,
                         "config_hash": expected_config, "source_hashes": source_fingerprints, "attempt_id": attempt_id,
                         "initial_params": [p.cpu() for p in initial_params],
                         "committed_log_chunks": raw_metrics.index, "log_lineage": lineage}, commit=on_checkpoint)
        checkpointing.atomic_json(os.path.join(raw, "checkpoint-timing", f"step{step}.json"),
                                  {"step": step, "seconds": time.monotonic() - save_started})


    def emit_wandb(values, step):
        nonlocal telemetry_failed_at, last_telemetry_check
        if not wb:
            return
        now = time.monotonic()
        try:
            wb.log({**values, "successful_update": step}, step=attempt)
            # log() queues asynchronously, so success alone cannot prove connectivity.
            if now - last_telemetry_check >= 60:
                import wandb
                last_telemetry_check = now
                wandb.Api(timeout=5).run(wb.path)
                telemetry_failed_at = None
        except Exception as error:
            telemetry_failed_at = telemetry_failed_at or now
            checkpointing.atomic_json(os.path.join(raw, "telemetry-failure.json"),
                                      {"step": step, "type": type(error).__name__, "message": str(error)})
        if telemetry_failed_at is not None and now - telemetry_failed_at >= 600:
            checkpoint(step)
            raise RuntimeError("W&B unavailable for ten minutes")

    def write(rec):
        rec["attempt_id"] = attempt_id
        rec["rollout_attempt"] = attempt
        raw_metrics.write(json.dumps(rec, allow_nan=False) + "\n")
        raw_metrics.flush()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        emit_wandb({k: v for k, v in rec.items() if isinstance(v, (int, float)) and k not in ("step", "type")}, rec["step"])

    def evaluate(step):
        t0 = time.time()
        with isolated_rng():
            ev, correct, records = greedy_eval(engine, model, eval_set, return_records=True)
        os.makedirs(os.path.join(out, "evals"), exist_ok=True)
        eval_record = {"records": records, "step": step, "n": len(correct), "correct": correct, "ids": [p.get("id", i) for i, p in enumerate(eval_set)]}
        checkpointing.atomic_json(os.path.join(raw, "evals", f"step{step}.json"), eval_record)
        checkpointing.atomic_json(os.path.join(out, "evals", f"step{step}.json"), eval_record)
        write({"type": "eval", "step": step, **ev, "t": time.time() - t0})
        print(f"[eval] step {step} acc={ev['eval_acc']:.3f} len={ev['eval_len']:.0f} ({time.time() - t0:.0f}s)")

    def passk(step, tag):
        t0 = time.time()
        with isolated_rng():
            pk = passk_eval(engine, model, passk_set, args.passk_k, args.temperature)
        write({"type": "passk", "step": step, "when": tag, **{f"test_{k}": v for k, v in pk.items()},
               "t": time.time() - t0})
        print(f"[pass@k] {tag}: " + " ".join(f"{k}={v:.3f}" for k, v in pk.items()))

    if not args.resume:
        model.save_pretrained(os.path.join(out, "adapters", "step0"))
        checkpoint(0)
    write({"type": "startup", "step": start, "seconds": time.monotonic() - startup})
    if start == 0:
        if args.eval_every:
            evaluate(0)
        if args.passk_n:
            passk(0, "start")

    step = start
    while step < args.steps:
        if guard:
            try:
                guard()
            except TimeoutError:
                checkpoint(step)
                for handle in (*files.values(), log, raw_metrics):
                    handle.close()
                if wb:
                    wb.finish()
                if on_checkpoint:
                    on_checkpoint()
                raise
        if attempt >= args.max_attempts or empty_attempts >= args.max_empty_attempts:
            checkpoint(step)
            raise RuntimeError("no-signal stop: rollout attempt limit reached")
        attempt += 1
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.eval()
        args.current_update = step + 1
        groups, st = sample_groups(engine, stream, train, args, attempt, files, reward_gen)
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
            "len_right": sum(s["length"] for s in s0 if s["correct"]) / sum(s["correct"] for s in s0) if any(s["correct"] for s in s0) else None,
            "len_wrong": sum(s["length"] for s in s0 if not s["correct"]) / sum(not s["correct"] for s in s0) if any(not s["correct"] for s in s0) else None,
            "len_finished": sum(fin) / len(fin) if fin else None,
            "len_min": min(s["length"] for s in s0), "len_max": max(s["length"] for s in s0),
            "trunc_frac": sum(s["truncated"] for s in s0) / len(s0),
            "n_right": nr,
            "questions_seen": stream.seen,
            "generated_tokens": st["generated_tokens"], "generated_answers": st["generated_answers"],
            "format_fraction": sum(s["formatted"] for s in s0) / len(s0),
            "shortfall_groups": args.questions - len(groups),
        })
        if not groups:
            empty_attempts += 1
            rec["type"] = "empty_attempt"
            write(rec)
            continue
        empty_attempts = 0
        if groups:
            args.current_update = step + 1
            model.train()
            t1 = time.time()
            rec.update(train_step(model, opt, sched, groups, args, device, tok.pad_token_id))
            rec["t_train"] = time.time() - t1
            step += 1
            rec["step"] = step
            params_now = [p for p in model.parameters() if p.requires_grad]
            rec["parameter_rms_from_initial"] = (sum((p.detach() - b).double().square().sum().item() for p, b in zip(params_now, initial_params)) / sum(p.numel() for p in params_now)) ** 0.5
            gap_strikes = gap_strikes + 1 if rec.get("gap_abs", 0) > 0.05 else 0
        t2 = time.time()
        engine.sync(step)                                           # next samples come from the new policy
        rec["t_sync"] = time.time() - t2
        elapsed = prev_elapsed + time.time() - t_start
        rec.update({"t_gen": t_gen, "t_step": time.time() - t0, "hours": elapsed / 3600,
                    "dollars": elapsed / 3600 * args.price_per_hour,
                    "vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None})
        write(rec)
        if wb:
            table_rows += [[step, qi, s["reward"], int(s["correct"]), s["length"], s["text"]]
                           for qi, s in st["table"][:args.table_per_step]]
        print(f"step {step:4d} | acc {rec['acc']:.3f} | wasted {rec['wasted_frac']:.2f} r{rec['rounds']} "
              f"| len {rec['len']:.0f} | kl {rec.get('kl', 0):.4f} | H {rec.get('entropy', 0):.3f}/"
              f"{rec.get('entropy_top', 0):.3f} | gn {rec.get('grad_norm', 0):.3f} "
              f"| {rec['t_gen']:.0f}s+{rec.get('t_train', 0):.0f}s | {rec['vram_gb']}GB", flush=True)
        for v in ("loss", "kl", "grad_norm"):
            assert v not in rec or rec[v] == rec[v], f"NaN in {v} at step {step}"

        if gap_strikes >= 3:
            checkpoint(step)
            raise RuntimeError("sampler gap exceeded 0.05 for three consecutive updates")
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            evaluate(step)
            if wb:
                import wandb
                emit_wandb({"samples": wandb.Table(columns=["step", "question", "reward", "correct", "length", "answer"],
                                               data=table_rows)}, step)
                table_rows.clear()
        if args.ckpt_every and (step % args.ckpt_every == 0 or step == args.steps):
            model.save_pretrained(os.path.join(out, "adapters", f"step{step}"))
            checkpoint(step)

    end_logged = any('"when": "end"' in l for l in open(os.path.join(out, "metrics.jsonl")))
    if args.passk_n and not end_logged:                  # also after a crash between the last checkpoint and here
        passk(args.steps, "end")
    model.save_pretrained(os.path.join(out, "adapter"))
    for f in (*files.values(), log, raw_metrics):
        f.close()
    if wb:
        wb.finish()
    if on_checkpoint:
        on_checkpoint()


if __name__ == "__main__":
    main()
