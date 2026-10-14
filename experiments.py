"""
Every Phase 3 experiment as a named entry: name -> the train_grpo.py flags that change.

    python experiments.py                         # list groups and experiments
    python experiments.py tier1 --platform modal  # print the exact commands of a group

The platform adds the settings that depend on the machine (engine, memory tricks, eval size).
Runs that get compared must share a platform (same engine + GPU type).

CHOSEN holds the decisions Stage B makes. Until B2 has fixed the LR, building any group except
the LR sweep fails on purpose, so nothing launches with a guessed LR.
"""
import argparse
import shlex

# Filled in after Stage B: B1 fixes the batch (questions x k), B2 the learning rate.
# 2026-09-30 speed test: 16 x 8 on L40S is cheapest per question. LR test (100 steps each, full test):
# 1e-5 50.8%, 2e-5 51.5%, 5e-5 52.5%, 1e-4 55.0% (start 47.7%); 1e-4 vs 5e-5 +2.6 [+0.1, +5.2].
CHOSEN = {"lr": "1e-4", "questions": 16, "k": 8, "steps": 200}

PLATFORMS = {
    # Big GPU: vLLM generates, no gradient checkpointing (saves one forward pass), full test every
    # eval. Micro-batch stays 8: without checkpointing, 8 answers x ~660 tokens already hold ~7 GB of
    # activations (~1.4 MB per token), which fits next to vLLM on every GPU in the B1 list.
    "modal": ["--engine", "vllm", "--no-grad-ckpt", "--micro-bs", "8", "--wandb"],
    # Laptop: HF generates; the full test takes 7.5 min per eval there, so 200 questions (+-3.5).
    "laptop": ["--engine", "hf", "--eval-n", "200", "--wandb"],
}

LR_SWEEP = {f"lr_{lr}": ["--lr", lr] for lr in ("1e-5", "2e-5", "5e-5", "1e-4")}

EXPERIMENTS = {
    **LR_SWEEP,
    # Tier 1: all compare to the baseline, so they can run together.
    "baseline_s0": [],
    "baseline_s1": ["--seed", "1"],
    "beta0": ["--beta", "0"],                                   # no KL leash
    "drgrpo": ["--no-std", "--agg", "const"],                   # Dr. GRPO recipe
    "token_agg": ["--agg", "token"],                            # DAPO averaging only
    "random_reward": ["--reward-mode", "random"],               # control: is the gain real?
    # Tier 2
    # Raw base model, plain prompt (the chat format doesn't suit it; see rl.data.build_prompt).
    "base_r1zero": ["--model", "Qwen/Qwen2.5-0.5B", "--prompt", "plain"],
    "base_r1zero_format": ["--model", "Qwen/Qwen2.5-0.5B", "--prompt", "plain", "--format-weight", "0.1"],
    "k4": ["--k", "4", "--questions", "{2q}"],                  # same answers per step, more questions
    "k16": ["--k", "16", "--questions", "{q/2}"],               # same answers per step, fewer questions
    "no_dynamic": ["--no-dynamic"],
    "oneshot": ["--overfit", "1", "--steps", "100"],
    "fourshot": ["--overfit", "4", "--steps", "100"],
    "entropy_bonus": ["--ent-coef", "0.005", "--ent-kind", "top"],   # coefficient set after Tier 1
    "random_reward_4upd": ["--reward-mode", "random", "--updates-per-batch", "4"],
    # Tier 3
    "rank1": ["--lora-r", "1", "--lora-alpha", "2"],            # alpha / r kept at 2
    "rank64": ["--lora-r", "64", "--lora-alpha", "128"],
}

GROUPS = {
    "lr_sweep": list(LR_SWEEP),
    "tier1": ["baseline_s0", "baseline_s1", "beta0", "drgrpo", "token_agg", "random_reward"],
    "tier2": ["base_r1zero", "base_r1zero_format", "k4", "k16", "no_dynamic", "oneshot", "fourshot",
              "random_reward_4upd"],
    "tier2_conditional": ["entropy_bonus"],                     # only if Tier 1's entropy rule says so
    "tier3": ["rank1", "rank64"],
}


def build(name, platform, lr=None, steps=None, extra=()):
    """-> (run name, argv list for train_grpo.main). lr/steps override CHOSEN (the sweep and smoke tests)."""
    lr = lr or CHOSEN["lr"]
    if lr is None and name not in LR_SWEEP:
        raise SystemExit(f"{name}: the LR is not chosen yet (Stage B2). Set CHOSEN['lr'] first.")
    q, k = CHOSEN["questions"], CHOSEN["k"]
    flags = [f.replace("{2q}", str(2 * q)).replace("{q/2}", str(q // 2)) for f in EXPERIMENTS[name]]
    # The sweep runs 100 steps at 16 x 8; at 64 x 8 pass steps=40 (plan B2).
    n_steps = steps or (100 if name in LR_SWEEP else CHOSEN["steps"])
    argv = ["--run", f"{name}_{platform}", *PLATFORMS[platform],
            "--questions", str(q), "--k", str(k), "--steps", str(n_steps)]
    if lr is not None and name not in LR_SWEEP:
        argv += ["--lr", str(lr)]
    argv += flags + list(extra)          # later flags win in argparse, so the experiment's own come last
    if steps:
        argv += ["--steps", str(steps)]  # ...except an explicit steps override, which always wins
    return f"{name}_{platform}", argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", nargs="?")
    ap.add_argument("--platform", default="modal", choices=list(PLATFORMS))
    args = ap.parse_args()
    if not args.group:
        for g, names in GROUPS.items():
            print(f"{g}: {', '.join(names)}")
        print(f"CHOSEN: {CHOSEN}")
        return
    for name in GROUPS.get(args.group, [args.group]):
        run, argv = build(name, args.platform)
        print("python train_grpo.py " + shlex.join(argv))


if __name__ == "__main__":
    main()
