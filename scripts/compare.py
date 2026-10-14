"""
One table over GRPO runs, with question-by-question (paired) comparisons.

    python scripts/compare.py beta0_modal drgrpo_modal --base baseline_s0_modal \
        --seeds baseline_s0_modal baseline_s1_modal
    python scripts/compare.py --all --plot

Why paired: two runs are graded on the SAME test questions, so compare them question by
question. Example with 5 questions: baseline right on [1,1,0,0,1], run B right on [1,1,1,0,1]:
B wins 1 question, loses 0, difference +20%. Resampling the questions (bootstrap) gives a 95%
range for that difference. It is tighter than the +-1.4 points of each run on its own, because
questions both runs get right (or both wrong) cancel out.

A difference is marked "real" only if its 95% range excludes 0 AND it is bigger than the gap
between the two baseline seeds (the luck-wobble of a whole run). Writes runs/grpo/compare.md.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "grpo")


def load(run):
    d = os.path.join(RUNS, run)
    rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    evals = sorted(glob.glob(os.path.join(d, "evals", "step*.json")),
                   key=lambda p: int(os.path.basename(p)[4:-5]))
    cfg = json.load(open(os.path.join(d, "config.json"))) if os.path.exists(os.path.join(d, "config.json")) else {}
    return {"cfg": cfg,
            "train": [r for r in rows if r["type"] == "train"],
            "eval": [r for r in rows if r["type"] == "eval"],
            "passk": [r for r in rows if r["type"] == "passk"],
            "correct": json.load(open(evals[-1]))["correct"] if evals else None}


def paired(a, b, n_boot=2000, seed=0):
    """Mean of (b - a) over questions and its 95% bootstrap range, in accuracy points."""
    d = torch.tensor(b, dtype=torch.float) - torch.tensor(a, dtype=torch.float)
    g = torch.Generator().manual_seed(seed)
    boots = d[torch.randint(len(d), (n_boot, len(d)), generator=g)].mean(1).sort().values
    return 100 * d.mean().item(), 100 * boots[int(0.025 * n_boot)].item(), 100 * boots[int(0.975 * n_boot)].item()


def tail(rows, key, n=10):
    vals = [r[key] for r in rows[-n:] if key in r]
    return sum(vals) / len(vals) if vals else None


def fmt(x, d=1, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--base", default=None, help="the run everything is compared to")
    ap.add_argument("--seeds", nargs=2, default=None, help="two baseline seeds: their gap = run-to-run luck")
    ap.add_argument("--plot", action="store_true", help="also draw the overlay with scripts/plot.py")
    ap.add_argument("--out", default=os.path.join(RUNS, "compare.md"))
    args = ap.parse_args()
    runs = args.runs or []
    if args.all:
        runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(RUNS, "*", "metrics.jsonl")))
    for r in [args.base, *(args.seeds or [])]:
        if r and r not in runs:
            runs.insert(0, r)
    data = {r: load(r) for r in runs}

    seed_gap = None
    lines = []
    if args.seeds and all(data[s]["correct"] for s in args.seeds):
        a, b = (data[s]["correct"] for s in args.seeds)
        m, lo, hi = paired(a, b)
        seed_gap = abs(m)
        lines.append(f"Seed gap ({args.seeds[1]} - {args.seeds[0]}): {m:+.1f} points (95% range {lo:+.1f} .. {hi:+.1f})\n")

    head = ["run", "engine", "steps", "test acc start", "test acc end", f"vs {args.base or 'base'} (95% range)",
            "verdict", "pass@k start", "pass@k end", "len right / wrong", "trunc", "KL", "entropy top first / last",
            "hours", "$"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r, d in data.items():
        ev, tr, pk = d["eval"], d["train"], {x["when"]: x for x in d["passk"]}
        vs, verdict = "-", "-"
        if args.base and r != args.base and d["correct"] and data[args.base]["correct"] \
                and len(d["correct"]) == len(data[args.base]["correct"]):
            m, lo, hi = paired(data[args.base]["correct"], d["correct"])
            vs = f"{m:+.1f} ({lo:+.1f} .. {hi:+.1f})"
            real = (lo > 0 or hi < 0) and (seed_gap is None or abs(m) > seed_gap)
            verdict = "real" if real else "noise"
            if seed_gap is None and real:
                verdict = "real? (no seed gap yet)"
        row = [r, d["cfg"].get("engine", "hf"), str(tr[-1]["step"]) if tr else "0",
               fmt(ev[0]["eval_acc"], pct=True) if ev else "-", fmt(ev[-1]["eval_acc"], pct=True) if ev else "-",
               vs, verdict,
               *(f"{fmt(pk.get(w, {}).get(f'test_pass@{kk}'), pct=True)} (k={kk})" if w in pk else "-"
                 for kk in [d["cfg"].get("passk_k", 8)] for w in ("start", "end")),
               f"{fmt(tail(tr, 'len_right'), 0)} / {fmt(tail(tr, 'len_wrong'), 0)}",
               fmt(tail(tr, "trunc_frac"), pct=True), fmt(tail(tr, "kl"), 4),
               f"{fmt(tr[0].get('entropy_top') if tr else None, 3)} / {fmt(tail(tr, 'entropy_top'), 3)}",
               fmt(tr[-1].get("hours") if tr else None, 2), fmt(tr[-1].get("dollars") if tr else None, 2)]
        lines.append("| " + " | ".join(row) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(args.out, "w") as f:
        f.write(text)
    if args.plot and runs:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "plot.py"), *runs,
                        "--out", os.path.join(RUNS, "compare.png")], cwd=ROOT)


if __name__ == "__main__":
    main()
