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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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
    ap.add_argument("--campaign-records", help="JSON mapping all seven model names to complete evaluation directories")
    ap.add_argument("--symbolic", action="store_true")
    ap.add_argument("--matched-gsm8k", help="JSON mapping model names to full greedy original-question outputs")
    args = ap.parse_args()
    if args.campaign_records:
        paths = json.load(open(args.campaign_records))
        from pathlib import Path
        from rl.checkpointing import file_hash
        models = {}
        for name, directory in paths.items():
            root = Path(directory)
            summary = json.loads((root/"summary.json").read_text())
            if summary.get("complete") is not True:
                raise ValueError("incomplete evaluation")
            rows = []
            for chunk in sorted((root/"chunks").glob("*.json")):
                if chunk.with_suffix(".sha256").read_text() != file_hash(chunk):
                    raise ValueError("evaluation checksum mismatch")
                rows.extend(json.loads(chunk.read_text()))
            expected_n = 5000 if args.symbolic else 1319
            if len(rows) != expected_n or len(rows) != summary["n"]:
                raise ValueError("missing evaluation records")
            models[name] = rows
        result = campaign_compare(models, symbolic=args.symbolic)
        if args.symbolic:
            if not args.matched_gsm8k:
                raise ValueError("Symbolic analysis requires matched GSM8K greedy outputs")
            greedy = {}
            for name, source in json.load(open(args.matched_gsm8k)).items():
                path = Path(source)
                if path.is_file():
                    greedy[name] = json.loads(path.read_text())["records"]
                else:
                    raw = [r for chunk in sorted((path/"chunks").glob("*.json")) for r in json.loads(chunk.read_text())]
                    greedy[name] = [{"id":r["id"], "question":r["question"], **r["samples"][0]} for r in raw]
            result["matched_originals"] = matched_symbolic(models, greedy)
        save_campaign_comparison(result, args.out)
        return
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





def campaign_compare(models, symbolic=False):
    """ID-aligned, per-model pass@k; question intervals conditional on these trained models."""
    import numpy as np
    from rl.campaign import RUNS
    from rl.eval import pass_at_k
    if set(models) != {"original", *RUNS}:
        raise ValueError("comparison requires original and all six final models")
    base = models["original"]
    ids = [r["id"] for r in base]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("duplicate or missing original IDs")
    n = len(base[0]["samples"])
    ks = [1] if symbolic else [k for k in (1,2,4,8,16,32,64) if k <= n]
    scores, table = {}, []
    for name, records in models.items():
        by_id = {r["id"]: r for r in records}
        if len(by_id) != len(records) or set(by_id) != set(ids):
            raise ValueError("missing, duplicate or extra comparison IDs")
        ordered = [by_id[q] for q in ids]
        for ref, row in zip(base, ordered):
            if row["question"] != ref["question"] or row["answer"] != ref["answer"]:
                raise ValueError("question/gold mismatch")
            if len(row["samples"]) != n or not 0 <= row["n_correct"] <= n:
                raise ValueError("sample count mismatch")
            if row["n_correct"] != sum(s["correct_strict"] for s in row["samples"]):
                raise ValueError("incorrect success count")
            if symbolic and row["template_id"] != ref["template_id"]:
                raise ValueError("template mismatch")
        scores[name] = {k: np.array([pass_at_k(n,r["n_correct"],k) for r in ordered]) for k in ks}
        for k in ks:
            table.append({"model": name, "k": k, "score": float(scores[name][k].mean())})
    arms = {}
    for arm in ("full", "positive", "negative"):
        arms[arm] = {k: (scores[arm+"_s0"][k] + scores[arm+"_s1"][k])/2 for k in ks}
        for k in ks:
            seed_values = [float(scores[arm+f"_s{s}"][k].mean()) for s in (0,1)]
            table.append({"model": arm+"_mean", "k": k, "score": sum(seed_values)/2,
                          "seed0": seed_values[0], "seed1": seed_values[1],
                          "seed_min": min(seed_values), "seed_max": max(seed_values)})
    units = np.array([r['template_id'] if symbolic else i for i,r in enumerate(base)])
    unique = np.unique(units)
    rng = np.random.default_rng(1234)
    draws = rng.integers(len(unique), size=(2000,len(unique)))
    intervals, changes = [], []
    comparisons = [("original", arm) for arm in arms] + [("positive","negative"),("full","positive"),("full","negative")]
    all_scores = {"original": scores["original"], **arms}
    for left,right in comparisons:
        for k in ks:
            delta = all_scores[right][k]-all_scores[left][k]
            # Templates have the same 50 instances; whole clusters keep within-template dependence.
            cluster_delta = np.array([delta[units==unit].mean() for unit in unique])
            boot = cluster_delta[draws].mean(1)
            low,high = np.quantile(boot,[.025,.975])
            intervals.append({"left":left,"right":right,"k":k,"difference":float(delta.mean()),
                              "low":float(low),"high":float(high),"bootstrap_draws":2000,"analysis_seed":1234})
    model_pairs = [('original',name) for name in scores if name!='original']
    model_pairs += [(left+f'_s{seed}',right+f'_s{seed}') for seed in (0,1)
                    for left,right in [('full','positive'),('full','negative'),('positive','negative')]]
    for left,right in model_pairs:
        for i,qid in enumerate(ids):
            a,b = scores[left][ks[-1]][i], scores[right][ks[-1]][i]
            changes.append({'left':left,'right':right,'id':qid,'gained_within_k':bool(a==0 and b>0),
                            'lost_within_k':bool(a>0 and b==0),
                            'success_rate_change':float(scores[right][1][i]-scores[left][1][i])})
    return {"table":table,"intervals":intervals,"question_changes":changes,
            "interval_scope":"Question-sampling intervals conditional on the trained models; not training-seed uncertainty or exhaustive finite-generation uncertainty.",
            "coverage_scope":f"Observed within {n} generations; zero successes does not establish impossibility.",
            "bootstrap_unit":"template" if symbolic else "question"}


def save_campaign_comparison(result, output):
    import csv
    from pathlib import Path
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path/"comparison.json").write_text(json.dumps(result,indent=2))
    for name in ("table","intervals","question_changes"):
        rows = result[name]
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (path/(name+".csv")).open("w") as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    if 'matched_originals' in result:
        rows=result['matched_originals']
        with (path/'matched_originals.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    from scripts.plot import plot_campaign_comparison
    plot_campaign_comparison(result,path/"passk.png")


def matched_symbolic(symbolic, greedy):
    if set(symbolic) != set(greedy):
        raise ValueError("matched model set differs")
    results = []
    for name, rows in symbolic.items():
        by_id = {r['id']:r for r in greedy[name]}
        if len(by_id)!=len(greedy[name]):
            raise ValueError("duplicate greedy IDs")
        templates = {}
        for row in rows:
            templates.setdefault(row['template_id'],[]).append(row)
        for template, instances in templates.items():
            original_ids={r['original_id'] for r in instances}
            if len(original_ids)!=1:
                raise ValueError("inconsistent template original")
            original_id=next(iter(original_ids))
            if original_id not in by_id:
                raise ValueError("missing matched original question")
            results.append({'model':name,'template_id':template,'original_id':original_id,
                            'original_correct':int(by_id[original_id]['correct_strict']),
                            'symbolic_accuracy':sum(r['n_correct']/len(r['samples']) for r in instances)/len(instances)})
    return results


if __name__ == "__main__":
    main()
