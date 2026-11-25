"""Final results of the six polarity runs: canonical histories, checks, tables and plots.

    python scripts/final_curves.py

Reads runs/final/training/<run>/ (downloaded from the Modal volume) and
runs/continuation/20261101/terminal.json. Writes:
    runs/final/curves/<run>.jsonl      one train record per update 1..200, one eval record per 0,25,..,200
    runs/final/greedy.md               greedy table at updates 0/50/100/150/200
    runs/final/greedy-changes.md       per-question right->wrong / wrong->right vs update 0
    runs/final/plots/*.png
Canonical = rebuilt only from the log chunks that terminal.json's log_lineage names
(attempt order as listed, first N chunks of each), then compared with the run's metrics.jsonl.
"""
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "runs" / "final"
TERMINAL = ROOT / "runs" / "continuation" / "20261101" / "terminal.json"
RUNS = ["full_s0", "full_s1", "positive_s0", "positive_s1", "negative_s0", "negative_s1"]
ARM_COLOR = {"full": "tab:blue", "positive": "tab:green", "negative": "tab:red"}
EVAL_STEPS = list(range(0, 201, 25))

# Greedy accuracy (%) from live metric snapshots taken during training; used to cross-check the rebuilt histories.
SNAPSHOT = {
    "full_s0": {0: 47.7, 25: 50.0, 50: 51.4, 75: 52.0, 100: 53.0},
    "full_s1": {0: 47.7, 25: 51.6, 50: 53.4, 75: 52.1, 100: 52.8},
    "positive_s0": {0: 47.7, 25: 49.5, 50: 51.6, 75: 51.7, 100: 53.5, 125: 51.9},
    "positive_s1": {0: 47.7, 25: 52.2, 50: 50.3, 75: 49.7, 100: 51.8},
    "negative_s0": {0: 47.7, 25: 47.4, 50: 48.4, 75: 51.7},
    "negative_s1": {0: 47.7, 25: 46.6, 50: 45.4, 75: 47.6, 100: 52.3},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def terminal_results():
    jobs = json.loads(TERMINAL.read_text())["state"]["jobs"]
    return {run: jobs[run]["result"] for run in RUNS}


def lineage_records(run_dir, lineage):
    """Records from the committed chunks only, in lineage order, checksum-verified."""
    out = []
    for attempt, counts in lineage.items():
        for i in range(counts["metrics"]):
            chunk = run_dir / "attempts" / attempt / "metrics" / f"{i:06d}.jsonl.gz"
            marker = json.loads(chunk.with_suffix(".complete.json").read_text())
            if sha256(chunk) != marker["sha256"]:
                raise ValueError(f"{chunk}: chunk checksum mismatch")
            with gzip.open(chunk, "rt") as f:
                out += [json.loads(line) for line in f if line.strip()]
    return out


def canonical_history(run, result):
    run_dir = FINAL / "training" / run
    from_chunks = lineage_records(run_dir, result["log_lineage"])
    from_file = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    train = [r for r in from_chunks if r["type"] == "train"]
    evals = [r for r in from_chunks if r["type"] == "eval"]
    if [r["step"] for r in train] != list(range(1, 201)):
        raise ValueError(f"{run}: train updates are not exactly 1..200 once each")
    if [r["step"] for r in evals] != EVAL_STEPS:
        raise ValueError(f"{run}: eval updates are not exactly 0,25,..,200 once each")
    return train + evals, {"chunks_equal_metrics_jsonl": from_chunks == from_file,
                           "records_from_chunks": len(from_chunks),
                           "types": {t: sum(r["type"] == t for r in from_chunks) for t in {r["type"] for r in from_chunks}},
                           "attempt_ids": sorted({r.get("attempt_id") for r in from_chunks}),
                           "lineage_attempts": list(result["log_lineage"])}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def build():
    results = terminal_results()
    (FINAL / "curves").mkdir(parents=True, exist_ok=True)
    hist, info, problems = {}, {}, []
    for run in RUNS:
        records, meta = canonical_history(run, results[run])
        with open(FINAL / "curves" / f"{run}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        hist[run] = records
        info[run] = meta
        # Snapshot cross-check, 0.1 points.
        ev = {r["step"]: 100 * r["eval_acc"] for r in records if r["type"] == "eval"}
        for step, want in SNAPSHOT[run].items():
            if abs(ev[step] - want) > 0.1 + 1e-9:
                problems.append(f"{run} update {step}: canonical {ev[step]:.2f} vs snapshot {want}")
    return hist, info, problems


def eval_file(run, step):
    p = FINAL / "training" / run / "evals" / f"step{step}.json"
    return json.loads(p.read_text()) if p.exists() else None


def greedy_table(hist):
    rows, lines, checks = {}, [], []
    for run in RUNS:
        ev = {r["step"]: r for r in hist[run] if r["type"] == "eval"}
        tr = {r["step"]: r for r in hist[run] if r["type"] == "train"}
        f200 = eval_file(run, 200)
        acc200_file = None
        if f200:
            recs = f200["records"]
            acc200_file = 100 * mean(r["correct_strict"] for r in recs)
            len_file = mean(r["length"] for r in recs)
            cut_file = mean(r["truncated"] for r in recs)
            checks.append((run, 100 * ev[200]["eval_acc"], acc200_file, ev[200]["eval_len"], len_file,
                           ev[200]["eval_trunc"], cut_file, f200["n"]))
        rows[run] = {"acc": {s: 100 * ev[s]["eval_acc"] for s in (0, 50, 100, 150, 200)},
                     "acc200_file": acc200_file,
                     "len200": ev[200]["eval_len"], "cut200": 100 * ev[200]["eval_trunc"],
                     "kl200": tr[200]["kl"], "kl_last10": mean(tr[s]["kl"] for s in range(191, 201))}
    head = ("| run | acc 0 | acc 50 | acc 100 | acc 150 | acc 200 | mean length 200 (tokens) "
            "| cutoff 200 (%) | KL at 200 | KL mean 191–200 |")
    lines += ["# Final greedy results (all 1,319 GSM8K test questions, strict scoring)", "",
              "Accuracy in %. Source: eval records of the canonical histories (runs/final/curves/); "
              "the update-200 accuracy is cross-checked against evals/step200.json. "
              "Length and cutoff are of the greedy answers at update 200; KL is the training-batch KL "
              "to the original model (k3 estimator, per token) logged at that update.", "",
              head, "|" + "---|" * 10]
    fmt = lambda r, name: (f"| {name} | " + " | ".join(f"{r['acc'][s]:.1f}" for s in (0, 50, 100, 150, 200))
                           + f" | {r['len200']:.0f} | {r['cut200']:.1f} | {r['kl200']:.4f} | {r['kl_last10']:.4f} |")
    for run in RUNS:
        lines.append(fmt(rows[run], run))
    lines += ["", "## Per arm: mean of the two seeds [seed range]", "",
              "| arm | acc 0 | acc 50 | acc 100 | acc 150 | acc 200 | mean length 200 | cutoff 200 (%) | KL at 200 | KL mean 191–200 |",
              "|" + "---|" * 10]
    for arm in ("full", "positive", "negative"):
        a, b = rows[arm + "_s0"], rows[arm + "_s1"]
        cell = lambda x, y, d: f"{(x + y) / 2:.{d}f} [{min(x, y):.{d}f}–{max(x, y):.{d}f}]"
        lines.append(f"| {arm} | " + " | ".join(cell(a['acc'][s], b['acc'][s], 1) for s in (0, 50, 100, 150, 200))
                     + f" | {cell(a['len200'], b['len200'], 0)} | {cell(a['cut200'], b['cut200'], 1)}"
                     + f" | {cell(a['kl200'], b['kl200'], 4)} | {cell(a['kl_last10'], b['kl_last10'], 4)} |")
    lines += ["", "## Cross-check: eval record at update 200 vs evals/step200.json", "",
              "| run | acc record | acc file | length record | length file | cutoff record | cutoff file | n |",
              "|" + "---|" * 8]
    for c in checks:
        lines.append(f"| {c[0]} | {c[1]:.2f} | {c[2]:.2f} | {c[3]:.1f} | {c[4]:.1f} | {100*c[5]:.2f} | {100*c[6]:.2f} | {c[7]} |")
    (FINAL / "greedy.md").write_text("\n".join(lines) + "\n")
    return rows, checks


def greedy_changes():
    """Per question, update 0 -> update 200, from each run's own evals/step{0,200}.json."""
    base, final, missing = {}, {}, []
    for run in RUNS:
        e0, e200 = eval_file(run, 0), eval_file(run, 200)
        if not e0 or not e200 or "records" not in e0 or "records" not in e200:
            missing.append(run)
            continue
        base[run] = {r["id"]: bool(r["correct_strict"]) for r in e0["records"]}
        final[run] = {r["id"]: bool(r["correct_strict"]) for r in e200["records"]}
    runs = [r for r in RUNS if r in base]
    # Are the six update-0 evaluations identical per question?
    ref = base[runs[0]]
    differ = {run: sum(base[run][q] != ref[q] for q in ref) for run in runs}
    lines = ["# Per-question greedy changes, update 0 -> update 200", "",
             "All 1,319 GSM8K test questions, greedy, strict scoring. Base = each run's own update-0 "
             "evaluation (evals/step0.json); final = evals/step200.json.", "",
             "Update-0 evaluations, questions whose 0/1 result differs from " + runs[0] + "'s: "
             + ", ".join(f"{r} {differ[r]}" for r in runs) + ".", "",
             "| run | right at 0 | right at 200 | right→wrong | wrong→right | net |", "|---|---|---|---|---|---|"]
    sets = {}
    for run in runs:
        lost = {q for q in base[run] if base[run][q] and not final[run][q]}
        gained = {q for q in base[run] if not base[run][q] and final[run][q]}
        sets[run] = (lost, gained)
        lines.append(f"| {run} | {sum(base[run].values())} | {sum(final[run].values())} | {len(lost)} | {len(gained)} | {len(gained) - len(lost):+d} |")
    lines += ["", "## Changes both seeds of an arm share", "",
              "Shared = the same question changed the same way in seed 0 and seed 1. "
              "'Expected by chance' = (count s0 × count s1) / 1,319 if the two seeds picked questions independently.", "",
              "| arm | right→wrong s0 / s1 | shared right→wrong (chance) | wrong→right s0 / s1 | shared wrong→right (chance) |",
              "|---|---|---|---|---|"]
    n = len(ref)
    for arm in ("full", "positive", "negative"):
        a, b = arm + "_s0", arm + "_s1"
        if a not in sets or b not in sets:
            lines.append(f"| {arm} | skipped: per-question records missing | | | |")
            continue
        (la, ga), (lb, gb) = sets[a], sets[b]
        lines.append(f"| {arm} | {len(la)} / {len(lb)} | {len(la & lb)} ({len(la)*len(lb)/n:.1f}) "
                     f"| {len(ga)} / {len(gb)} | {len(ga & gb)} ({len(ga)*len(gb)/n:.1f}) |")
    if missing:
        lines += ["", "Skipped (no per-question records at update 0 or 200): " + ", ".join(missing)]
    (FINAL / "greedy-changes.md").write_text("\n".join(lines) + "\n")
    return sets, differ, missing


def rolling(xs, n=10):
    return [mean(xs[max(0, i - n + 1): i + 1]) for i in range(len(xs))]


def plots(hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = FINAL / "plots"
    out.mkdir(parents=True, exist_ok=True)

    def style(run):
        arm, seed = run.rsplit("_s", 1)
        return dict(color=ARM_COLOR[arm], linestyle="-" if seed == "0" else "--", label=run, linewidth=1.4)

    def series(run, kind, key):
        rs = [r for r in hist[run] if r["type"] == kind and r.get(key) is not None]
        return [r["step"] for r in rs], [r[key] for r in rs]

    # 1. greedy accuracy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in RUNS:
        x, y = series(run, "eval", "eval_acc")
        ax.plot(x, [100 * v for v in y], marker="o", markersize=3, **style(run))
    ax.set(xlabel="update", ylabel="greedy accuracy, 1,319 test questions (%)",
           title="Greedy test accuracy (strict)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "greedy_accuracy.png", dpi=150); plt.close(fig)

    # 2. training reward
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in RUNS:
        x, y = series(run, "train", "reward")
        ax.plot(x, rolling(y), **style(run))
    ax.set(xlabel="update", ylabel="mean reward per sampled answer (rolling mean, 10 updates)",
           title="Training reward (fraction of sampled answers right)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "train_reward.png", dpi=150); plt.close(fig)

    # 3. length and cutoff (training samples, rolling 10; greedy eval points as markers)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in RUNS:
        x, y = series(run, "train", "len")
        axes[0].plot(x, rolling(y), **style(run))
        x, y = series(run, "train", "trunc_frac")
        axes[1].plot(x, [100 * v for v in rolling(y)], **style(run))
    axes[0].set(xlabel="update", ylabel="mean sampled answer length (tokens, rolling 10)",
                title="Answer length (training samples)")
    axes[1].set(xlabel="update", ylabel="answers cut off at 512 tokens (%, rolling 10)",
                title="Cutoff fraction (training samples)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "length_cutoff.png", dpi=150); plt.close(fig)

    # 3b. same for the greedy evals
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in RUNS:
        x, y = series(run, "eval", "eval_len")
        axes[0].plot(x, y, marker="o", markersize=3, **style(run))
        x, y = series(run, "eval", "eval_trunc")
        axes[1].plot(x, [100 * v for v in y], marker="o", markersize=3, **style(run))
    axes[0].set(xlabel="update", ylabel="mean greedy answer length (tokens)", title="Answer length (greedy eval)")
    axes[1].set(xlabel="update", ylabel="greedy answers cut off at 512 tokens (%)", title="Cutoff fraction (greedy eval)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "length_cutoff_greedy.png", dpi=150); plt.close(fig)

    # 4. KL and entropy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in RUNS:
        x, y = series(run, "train", "kl")
        axes[0].plot(x, rolling(y), **style(run))
        x, y = series(run, "train", "entropy")
        axes[1].plot(x, rolling(y), **style(run))
    axes[0].set(xlabel="update", ylabel="KL to original model (nats per token, rolling 10)",
                title="KL (k3 estimate on training samples)")
    axes[1].set(xlabel="update", ylabel="entropy (nats per token, rolling 10)",
                title="Entropy of the next-token distribution on training samples")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "kl_entropy.png", dpi=150); plt.close(fig)
    return sorted(p.name for p in out.glob("*.png"))


def main():
    hist, info, problems = build()
    print(json.dumps(info, indent=1))
    if problems:
        print("SNAPSHOT MISMATCH:\n" + "\n".join(problems))
        sys.exit(1)
    rows, checks = greedy_table(hist)
    sets, differ, missing = greedy_changes()
    print("plots:", plots(hist))
    print((FINAL / "greedy.md").read_text())
    print((FINAL / "greedy-changes.md").read_text())


if __name__ == "__main__":
    main()
