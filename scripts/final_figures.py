"""Make the README figures from the final evaluation outputs.

Reads (all written by scripts/final_eval_links.py, scripts/compare.py and scripts/final_spread.py):
  runs/final/eval/<model>/gsm8k-k64/summary.json   pass@k per model
  runs/final/spread.md                             how many of the 64 tries are right, per question
  runs/final/curves/<run>.jsonl                    canonical training histories (scripts/final_curves.py)
Writes docs/figures/{passk.png, buckets.png, greedy_accuracy.png}, all in the same colors.

Usage: python scripts/final_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "runs" / "final"
OUT = ROOT / "docs" / "figures"

MODELS = ["original", "full_s0", "full_s1", "positive_s0", "positive_s1", "negative_s0", "negative_s1"]
KS = [1, 2, 4, 8, 16, 32, 64]
COLOR = {"original": "#7a7974", "full": "#2a78d6", "positive": "#eb6834", "negative": "#1baf7a"}
ARM_NAME = {"full": "Full GRPO", "positive": "Positive-only", "negative": "Negative-only"}
BUCKETS = ["0/64", "1-3/64", "4-16/64", "17-48/64", "49-63/64", "64/64"]
GRAYS = ["#c9c7c0", "#a6a49c", "#83817a", "#615f59", "#403f3b", "#22211f"]


def name(model):
    if model == "original":
        return "Original"
    arm, seed = model.split("_")
    return f"{ARM_NAME[arm]} · seed {seed[1]}"


def color(model):
    return COLOR["original"] if model == "original" else COLOR[model.split("_")[0]]


def load_passk():
    result = {}
    for model in MODELS:
        summary = json.loads((FINAL / "eval" / model / "gsm8k-k64" / "summary.json").read_text())
        result[model] = [100 * summary["passk"][f"pass@{k}"] for k in KS]
    return result


def load_buckets():
    # the first table in spread.md: model | pass@1 | pass@32 | pass@64 | then the six bucket shares
    rows = {}
    for line in (FINAL / "spread.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 10 and cells[0] in MODELS and cells[0] not in rows:
            rows[cells[0]] = [float(c.rstrip("%")) for c in cells[4:]]
    missing = [m for m in MODELS if m not in rows]
    if missing:
        raise SystemExit(f"bucket rows missing in spread.md: {missing}")
    return rows


def plot_passk(passk):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for model in MODELS:
        dashed = model.endswith("_s1")
        ax.plot(range(len(KS)), passk[model], color=color(model), lw=2.5 if model == "original" else 2,
                ls="--" if dashed else "-", marker="o", ms=3.5, label=name(model))
    ax.set_xticks(range(len(KS)), [str(k) for k in KS])
    ax.set_xlabel("number of tries k (temperature 1)")
    ax.set_ylabel("solved at least once (%)")
    ax.set_title("GSM8K test (1,319 questions): pass@k after 200 GRPO updates")
    ax.set_ylim(30, 100)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "passk.png")
    plt.close(fig)


def plot_buckets(buckets):
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    for row, model in enumerate(reversed(MODELS)):
        left = 0.0
        for i, share in enumerate(buckets[model]):
            ax.barh(row, share, left=left, color=GRAYS[i], edgecolor="white", linewidth=1.5,
                    label=BUCKETS[i] if row == 0 else None)
            left += share
        ax.plot([-1.2], [row], marker="s", ms=6, color=color(model), clip_on=False)
    ax.set_yticks(range(len(MODELS)), [name(m) for m in reversed(MODELS)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of questions (%)")
    ax.set_title("How many of the 64 tries are right, per question")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.18))
    fig.tight_layout()
    fig.savefig(OUT / "buckets.png")
    plt.close(fig)


def load_greedy():
    # eval records (greedy accuracy on all 1,319 test questions) every 25 updates
    result = {}
    for model in MODELS[1:]:
        rows = [json.loads(line) for line in (FINAL / "curves" / f"{model}.jsonl").read_text().splitlines() if line.strip()]
        evals = sorted((r["step"], 100 * r["eval_acc"]) for r in rows if r.get("type") == "eval")
        result[model] = evals
    return result


def plot_greedy(greedy):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for model, evals in greedy.items():
        steps = [s for s, _ in evals]
        accs = [a for _, a in evals]
        ax.plot(steps, accs, color=color(model), lw=2, ls="--" if model.endswith("_s1") else "-",
                marker="o", ms=3.5, label=name(model))
    ax.set_xticks(range(0, 201, 25))
    ax.set_xlabel("update")
    ax.set_ylabel("greedy accuracy (%)")
    ax.set_title("Greedy accuracy on GSM8K test (1,319 questions) during training")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "greedy_accuracy.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_passk(load_passk())
    plot_buckets(load_buckets())
    plot_greedy(load_greedy())
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
