"""Answer-spread analysis of the final 64-tries GSM8K evaluation (local, plain Python).

    "<venv python>" scripts/final_spread.py [--root runs/final/eval] [--out runs/final/spread.md]

Reads every <root>/<model>/gsm8k-k64/chunks/*.json written by scripts/final_eval.py. Each record is one
question {id, answer (gold), samples: [64 x {pred, source, correct_strict, truncated, ...}], n_correct}.

Per model:
- pass@1 / pass@32 / pass@64: unbiased estimator 1 - C(n-c, k) / C(n, k) (same formula as
  rl.eval.pass_at_k, copied here so this script does not import torch). pass@64 with n = 64 is the
  share of questions with at least 1 of 64 right.
- right-out-of-64 buckets: 0, 1-3, 4-16, 17-48, 49-63, 64 (share of questions).
- Final answer of a sample: the scorer's stored `pred` (re-extracted with rl.reward.parse_answer, the
  function strict scoring uses, only if a sample has no `pred`). Main definition, matching strict
  scoring: a sample that is cut off (truncated), has no answer, or whose answer is not formatted
  (source not boxed/hash) gets the single value "no answer"; otherwise its value is pred rounded to
  4 decimals (is_correct's tolerance is 1e-4).
  - distinct answers: mean number of different values among the 64.
  - top answer share: mean (count of the most common value) / 64.
  - maj@64: share of questions whose most common value is right ("no answer" is never right). On a tie
    for most common, the question gets the fraction of tied values that are right.
  A second table repeats these three with unformatted final numbers counted as answers (source "last"
  included), to show how much the "formatted only" rule matters.
- gained / lost vs the original: lost = original right at least once in 64, model never;
  gained = the reverse. Per arm, the questions both seeds lose (or gain), and the overlap expected by
  chance n0 * n1 / N, N = questions eligible (lost: original solved; gained: original never solved).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rl.reward import is_correct, parse_answer  # noqa: E402

K = 64
NO_ANSWER = "no answer"
BUCKETS = [("0", 0, 0), ("1-3", 1, 3), ("4-16", 4, 16), ("17-48", 17, 48), ("49-63", 49, 63), ("64", 64, 64)]
ORDER = ["original", "full_s0", "full_s1", "positive_s0", "positive_s1", "negative_s0", "negative_s1"]
ARMS = ["full", "positive", "negative"]


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    p_all_wrong = 1.0
    for i in range(k):
        p_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - p_all_wrong


def load_records(directory):
    """All question records of one model, by question id."""
    chunks = sorted((Path(directory) / "chunks").glob("*.json"))
    if not chunks:
        raise FileNotFoundError(f"no chunks in {directory}/chunks")
    records = {}
    for p in chunks:
        for r in json.loads(p.read_text()):
            if r["id"] in records:
                raise ValueError(f"duplicate question id {r['id']} in {p}")
            if len(r["samples"]) != K:
                raise ValueError(f"{p}: question {r['id']} has {len(r['samples'])} samples, expected {K}")
            if r["n_correct"] != sum(bool(s["correct_strict"]) for s in r["samples"]):
                raise ValueError(f"{p}: question {r['id']}: n_correct disagrees with the samples")
            records[r["id"]] = r
    return records


def answer_value(sample, formatted_only=True):
    if "pred" in sample:
        pred, source = sample["pred"], sample["source"]
    else:
        pred, source = parse_answer(sample["text"])
    if sample.get("truncated") or pred is None:
        return NO_ANSWER
    if formatted_only and source not in ("boxed", "hash"):
        return NO_ANSWER
    return round(float(pred), 4)


def spread(record, formatted_only=True):
    """(number of distinct values, top share, majority credit) for one question."""
    counts = Counter(answer_value(s, formatted_only) for s in record["samples"])
    top = max(counts.values())
    tied = [v for v, c in counts.items() if c == top]
    right = sum(v != NO_ANSWER and is_correct(v, record["answer"]) for v in tied)
    return len(counts), top / K, right / len(tied)


def model_stats(records):
    n = len(records)
    counts = [r["n_correct"] for r in records.values()]
    st = {"n": n}
    for k in (1, 32, 64):
        st[f"pass@{k}"] = sum(pass_at_k(K, c, k) for c in counts) / n
    for name, lo, hi in BUCKETS:
        st[f"b{name}"] = sum(lo <= c <= hi for c in counts) / n
    for tag, formatted_only in (("", True), ("_any", False)):
        sp = [spread(r, formatted_only) for r in records.values()]
        st["distinct" + tag] = sum(x[0] for x in sp) / n
        st["top" + tag] = sum(x[1] for x in sp) / n
        st["maj" + tag] = sum(x[2] for x in sp) / n
    st["ids"] = set(records)
    st["solved"] = {i for i, r in records.items() if r["n_correct"] > 0}
    return st


def pct(x):
    return f"{100 * x:.1f}%"


def render(stats):
    models = [m for m in ORDER if m in stats] + sorted(m for m in stats if m not in ORDER)
    n_q = stats[models[0]]["n"]
    out = ["# Answer spread, GSM8K test, 64 tries per question (temperature 1)", "",
           f"{n_q} questions per model. Strict scoring: an answer counts only inside \\boxed{{}} or after ####. "
           "Definitions in the docstring of scripts/final_spread.py.",
           "", "## Pass rates, and how many of the 64 tries are right (share of questions)", "",
           "| model | pass@1 | pass@32 | pass@64 | " + " | ".join(f"{b[0]}/64" for b in BUCKETS) + " |",
           "|---" * (4 + len(BUCKETS)) + "|"]
    for m in models:
        s = stats[m]
        out.append(f"| {m} | {pct(s['pass@1'])} | {pct(s['pass@32'])} | {pct(s['pass@64'])} | "
                   + " | ".join(pct(s[f'b{b[0]}']) for b in BUCKETS) + " |")
    for title, tag in (("Spread of final answers (cut off, missing or unformatted = one \"no answer\" value)", ""),
                       ("Same, with unformatted final numbers counted as answers (cut off or missing = \"no answer\")", "_any")):
        out += ["", f"## {title}", "", "| model | distinct answers | top answer share | maj@64 |", "|---|---|---|---|"]
        for m in models:
            s = stats[m]
            out.append(f"| {m} | {s['distinct' + tag]:.2f} | {pct(s['top' + tag])} | {pct(s['maj' + tag])} |")
    if "original" in stats:
        base = stats["original"]["solved"]
        n_unsolved = n_q - len(base)
        out += ["", "## Gained and lost vs the original (solved = right at least once in 64)", "",
                f"The original solves {len(base)} of {n_q}; {n_unsolved} never.", "",
                "| model | lost | gained | net |", "|---|---|---|---|"]
        lost, gained = {}, {}
        for m in models:
            if m == "original":
                continue
            lost[m] = base - stats[m]["solved"]
            gained[m] = stats[m]["solved"] - base
            out.append(f"| {m} | {len(lost[m])} | {len(gained[m])} | {len(gained[m]) - len(lost[m]):+d} |")
        rows = []
        for arm in ARMS:
            a, b = f"{arm}_s0", f"{arm}_s1"
            if a in lost and b in lost:
                for kind, sets, N in (("lost", lost, len(base)), ("gained", gained, n_unsolved)):
                    n0, n1, both = len(sets[a]), len(sets[b]), len(sets[a] & sets[b])
                    chance = f"{n0 * n1 / N:.1f}" if N else "n/a"
                    rows.append(f"| {arm} | {kind} | {n0} | {n1} | {both} | {chance} | {N} |")
        if rows:
            out += ["", "Seed agreement: questions that both seeds of an arm lose (or gain), next to the overlap "
                    "expected if each seed's set were drawn at random from the eligible questions (n0 x n1 / N).", "",
                    "| arm | kind | seed 0 | seed 1 | both | by chance | N eligible |", "|---|---|---|---|---|---|---|"]
            out += rows
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "runs" / "final" / "eval"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "final" / "spread.md"))
    args = ap.parse_args(argv)
    dirs = sorted(Path(args.root).glob(f"*/gsm8k-k{K}"))
    if not dirs:
        raise SystemExit(f"no */gsm8k-k{K} directories under {args.root}")
    stats = {d.parent.name: model_stats(load_records(d)) for d in dirs}
    ids = {m: frozenset(s["ids"]) for m, s in stats.items()}
    if len(set(ids.values())) != 1:
        raise SystemExit(f"models cover different questions: { {m: len(i) for m, i in ids.items()} }")
    text = render(stats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
