"""
Run experiments one after another on the laptop (e.g. overnight).

    python scripts/run_queue.py baseline_s0 beta0          # names or group names from experiments.py
    python scripts/run_queue.py tier1 --lr 2e-5 --dry      # print the commands only

Each run is its own process, so GPU memory is fully freed between runs. A run that already has
metrics is continued with --resume; a finished run is skipped. A failing run is logged and the
queue moves on. Output of each run: runs/grpo/<run>.log
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import GROUPS, build  # noqa: E402
from train_grpo import parse_args  # noqa: E402


def last_step(out):
    p = os.path.join(out, "metrics.jsonl")
    if not os.path.exists(p):
        return None
    steps = [json.loads(l)["step"] for l in open(p) if '"train"' in l]
    return max(steps, default=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--lr", default=None, help="override experiments.CHOSEN['lr']")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    names = [n for x in args.names for n in GROUPS.get(x, [x])]

    for name in names:
        run, argv = build(name, "laptop", lr=args.lr, steps=args.steps)
        out = os.path.join(ROOT, "runs", "grpo", run)
        steps = parse_args(argv).steps
        done = last_step(out)
        if done is not None and done >= steps and os.path.exists(os.path.join(out, "adapter")):
            print(f"[queue] {run}: finished, skipping")
            continue
        if done is not None:
            argv.append("--resume")
        cmd = [sys.executable, os.path.join(ROOT, "train_grpo.py"), *argv]
        print(f"[queue] {run}: {' '.join(cmd[1:])}", flush=True)
        if args.dry:
            continue
        t0 = time.time()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".log", "a") as log:
            code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"[queue] {run}: exit {code} after {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
