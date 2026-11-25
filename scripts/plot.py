"""
Plot one or more GRPO runs from runs/grpo/<run>/metrics.jsonl.

    python scripts/plot.py baseline                 # -> runs/grpo/baseline/plots.png
    python scripts/plot.py baseline beta0 --out runs/grpo/compare.png

Thin lines are raw per-step values; thick lines are a rolling mean (--smooth steps).
"""
import argparse
import json
import os
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

PANELS = [  # (title, [(key, label)], y-limits or None)
    ("Train accuracy (round-0 samples)", [("acc", None)], (0, 1)),
    ("Greedy test accuracy (eval_acc; full test unless --eval-n)", [("eval_acc", None)], None),
    ("Wasted groups (all right / all wrong)", [("wasted_frac", "wasted"), ("all_right_frac", "all right"),
                                              ("all_wrong_frac", "all wrong")], (0, 1)),
    ("Answer length (tokens)", [("len", "all"), ("len_right", "right"), ("len_wrong", "wrong")], None),
    ("Truncated at max tokens", [("trunc_frac", None)], (0, None)),
    ("KL to the original model", [("kl", None)], (0, None)),
    ("Entropy per token", [("entropy", "full"), ("entropy_top", "top-p")], (0, None)),
    ("Grad norm (before clipping)", [("grad_norm", None)], (0, None)),
    ("Seconds per step", [("t_gen", "sampling"), ("t_train", "training")], (0, None)),
]


def load(run):
    rows = [json.loads(l) for l in open(os.path.join("runs", "grpo", run, "metrics.jsonl"))]
    return [r for r in rows if r["type"] == "train"], [r for r in rows if r["type"] == "eval"]


def rolling(y, w):
    out, acc = [], []
    for v in y:
        acc.append(v)
        acc = acc[-w:]
        out.append(sum(acc) / len(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--campaign-root", help="campaign folder containing training/<logical-run>/metrics.jsonl")
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.campaign_root:
        plot_campaign_training(args.campaign_root, args.out)
        return
    if not args.runs:
        ap.error("runs or --campaign-root required")
    data = {r: load(r) for r in args.runs}
    multi = len(args.runs) > 1

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (title, series, ylim) in zip(axes.flat, PANELS):
        for ri, run in enumerate(args.runs):
            train, evals = data[run]
            for si, (key, label) in enumerate(series):
                rows = evals if key.startswith("eval") else train
                pts = [(r["step"], r[key]) for r in rows if key in r and r[key] is not None]
                if not pts:
                    continue
                x, y = zip(*pts)
                c = COLORS[ri if multi else si]
                ls = ["-", "--", ":"][si] if multi else "-"
                name = " · ".join(n for n in (run if multi else None, label) if n)
                if key.startswith("eval"):
                    ax.plot(x, y, marker="o", ms=4, lw=2, color=c, ls=ls, label=name or None)
                else:
                    ax.plot(x, y, lw=0.8, alpha=0.3, color=c, ls=ls)
                    ax.plot(x, rolling(y, args.smooth), lw=2, color=c, ls=ls, label=name or None)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step", fontsize=9)
        ax.grid(alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False)
        elif not ax.lines:
            ax.text(0.5, 0.5, "not logged in this run", ha="center", va="center",
                    transform=ax.transAxes, color="#8a8984")
    fig.suptitle(" vs ".join(args.runs), fontsize=13)
    fig.tight_layout()
    out = args.out or os.path.join("runs", "grpo", args.runs[0], "plots.png")
    fig.savefig(out, dpi=110)
    print(out)





def plot_campaign_comparison(result, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(dict.fromkeys(row['model'] for row in result['table'] if not row['model'].endswith('_mean')))
    for model in models:
        rows = [r for r in result['table'] if r['model'] == model]
        ax.plot([r['k'] for r in rows],[r['score'] for r in rows],marker='o',label=model,
                linestyle='--' if model.endswith('_s1') else '-')
    ax.set(xlabel='Number of attempts (k)',ylabel='Pass@k',ylim=(0,1))
    ax.set_xscale('log',base=2)
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output,dpi=160);plt.close(fig)


def plot_campaign_training(root, output=None):
    from pathlib import Path
    # The canonical metrics views are rebuilt from checkpoint lineage on resume.
    from rl.campaign import RUNS
    panels = [('eval_acc','Greedy accuracy'),('entropy','Full entropy'),('entropy_top','Top-90% entropy'),
              ('objective_kl','Objective-weighted KL'),('parameter_rms_change','RMS parameter movement'),
              ('parameter_rms_from_initial','RMS displacement from initialization'),
              ('policy_grad_norm','Policy gradient norm'),('weighted_kl_grad_norm','Weighted KL gradient norm'),
              ('policy_kl_alignment','Policy/KL gradient alignment'),('grad_norm','Pre-clip gradient norm'),
              ('gap_abs','Absolute sampler/trainer gap'),('t_step','Seconds per update')]
    fig, axes=plt.subplots(4,3,figsize=(15,15))
    for i,run in enumerate(RUNS):
        path=Path(root)/'training'/run/'metrics.jsonl'
        rows=[json.loads(line) for line in path.open()]
        for ax,(key,title) in zip(axes.flat,panels):
            pts=[(r['step'],r[key]) for r in rows if r.get(key) is not None]
            if pts:ax.plot(*zip(*pts),color=COLORS[i%3],linestyle='--' if run.endswith('_s1') else '-',label=run)
            ax.set(title=title,xlabel='Successful update');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output or str(Path(root)/'training-trajectories.png'),dpi=160);plt.close(fig)


if __name__ == "__main__":
    main()
