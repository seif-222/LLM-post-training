"""After the paid final evaluation: download its outputs and write the inputs of scripts/compare.py.

    "<venv python>" scripts/final_eval_links.py --k 64            # download, then write the JSON files
    "<venv python>" scripts/final_eval_links.py --k 64 --no-download

Downloads /final-eval/20261123/ of Volume lm-rl to runs/final/eval/ (a volume read; starts no
function) and writes, in runs/final/eval/:
    campaign-gsm8k.json      model -> <target>/gsm8k-k<k>   (sampled GSM8K)
    campaign-symbolic.json   model -> <target>/symbolic      (greedy GSM-Symbolic)
    matched-gsm8k.json       original -> original/gsm8k-greedy (directory of chunks);
                             trained run -> runs/final/training/<run>/evals/step200.json
compare.py's matched loader reads a file as {"records": [...]} with id and correct_strict per
record, which is the format of evals/step200.json, so those are used as they are.
Then:
    python scripts/compare.py --campaign-records runs/final/eval/campaign-gsm8k.json --out runs/final/compare-gsm8k
    python scripts/compare.py --campaign-records runs/final/eval/campaign-symbolic.json --symbolic \
        --matched-gsm8k runs/final/eval/matched-gsm8k.json --out runs/final/compare-symbolic
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/final-eval/20261123"
TRAINED = ["full_s0", "full_s1", "positive_s0", "positive_s1", "negative_s0", "negative_s1"]
TARGETS = ["original", *TRAINED]


def download(local_root):
    import modal
    vol = modal.Volume.from_name("lm-rl")
    files = [e for e in vol.listdir(REMOTE, recursive=True) if e.type == modal.volume.FileEntryType.FILE]

    def get(e):
        rel = e.path.lstrip("/").split(REMOTE.lstrip("/") + "/", 1)[1]
        dst = Path(local_root) / rel
        if dst.exists() and dst.stat().st_size == e.size:
            return e.size
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        with open(tmp, "wb") as f:
            vol.read_file_into_fileobj(e.path, f)
        if tmp.stat().st_size != e.size:
            raise IOError(f"short download: {e.path}")
        os.replace(tmp, dst)
        return e.size

    with ThreadPoolExecutor(16) as ex:
        total = sum(ex.map(get, files))
    print(f"downloaded {len(files)} files, {total / 2**20:.1f} MiB -> {local_root}")


def links(local_root, k, training_root):
    local_root, training_root = Path(local_root), Path(training_root)
    gsm, sym, matched, problems = {}, {}, {}, []
    for t in TARGETS:
        gsm[t] = str(local_root / t / f"gsm8k-k{k}")
        sym[t] = str(local_root / t / "symbolic")
        matched[t] = str(local_root / t / "gsm8k-greedy") if t == "original" \
            else str(training_root / t / "evals" / "step200.json")
        for d in (gsm[t], sym[t]):
            s = Path(d) / "summary.json"
            if not s.exists() or json.loads(s.read_text()).get("complete") is not True:
                problems.append(f"incomplete: {d}")
        m = Path(matched[t])
        if t == "original":
            if not (m / "summary.json").exists():
                problems.append(f"incomplete: {m}")
        else:
            recs = json.loads(m.read_text()).get("records") if m.exists() else None
            if not recs or len(recs) != 1319 or not all("id" in r and "correct_strict" in r for r in recs):
                problems.append(f"step200 records missing or not in compare.py's matched format: {m}")
    out = {"campaign-gsm8k.json": gsm, "campaign-symbolic.json": sym, "matched-gsm8k.json": matched}
    local_root.mkdir(parents=True, exist_ok=True)
    for name, mapping in out.items():
        (local_root / name).write_text(json.dumps(mapping, indent=1) + "\n")
    return out, problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, choices=(32, 64))
    ap.add_argument("--out", default=str(ROOT / "runs" / "final" / "eval"))
    ap.add_argument("--training", default=str(ROOT / "runs" / "final" / "training"))
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args(argv)
    if not args.no_download:
        download(args.out)
    out, problems = links(args.out, args.k, args.training)
    for name in out:
        print("wrote", Path(args.out) / name)
    if problems:
        print("PROBLEMS:\n" + "\n".join(problems))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
