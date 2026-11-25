"""Final sampled evaluation: the original model and the six update-200 adapters.

    "<venv python>" scripts/final_eval.py --k 64 --dry-run      # prints the plan; never contacts Modal
    "<venv python>" -m modal run scripts/final_eval.py --k 64   # PAID: seven L40S calls

Use the plain-python form for the dry run: `modal run` creates the app and resolves its image
(building it if not cached) before the entrypoint runs, so `modal run ... --dry-run` is not free.

Per target, written under /vol/final-eval/20261123/<target>/ (Volume lm-rl):
  1. gsm8k-k<k>/     all 1,319 GSM8K test questions, k samples each, temperature 1.0
  2. symbolic/       GSM-Symbolic main, 5,000 questions, greedy
  3. gsm8k-greedy/   original only: GSM8K greedy (the matched-Symbolic comparison needs it;
                     the trained runs use their own evals/step200.json)
Everything goes through rl.evaluation_io.evaluate_manifest: chunks of 32 questions, the volume
is committed after each chunk, and finished chunks are skipped on a re-run. So a re-run of the
same command resumes. Seven independent calls; no controller, ledger or ownership records.

The checkpoint of each trained target comes from runs/continuation/20261101/terminal.json and
is passed to the remote function; the function re-hashes the adapter there and refuses a mismatch.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modal

TERMINAL = ROOT / "runs" / "continuation" / "20261101" / "terminal.json"
MANIFESTS = {"gsm8k": str(ROOT / "runs" / "preparation" / "001" / "gsm8k-test-manifest.json"),
             "symbolic": str(ROOT / "runs" / "preparation" / "001" / "symbolic-test-manifest.json")}
TRAINED = ["full_s0", "full_s1", "positive_s0", "positive_s1", "negative_s0", "negative_s1"]
TARGETS = ["original", *TRAINED]
VOL = "/vol"
OUT_ROOT = f"{VOL}/final-eval/20261123"
CODE = "/root/lm_rl"                       # where campaign_image puts the project code
TIMEOUT_S = 3 * 3600                       # Modal function timeout
GUARD_MARGIN_S = 15 * 60                   # stop starting new chunks this long before the timeout
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")

# Cost model. Seconds measured for 256 questions x 64 samples on L40S.
MEASURED_256x64_S = {"original": 304.8, "trained": 428.6}
GSM8K_N, SYMBOLIC_N = 1319, 5000
SYMBOLIC_GREEDY_S = 120                    # ~2 min per model for 5,000 greedy questions
STARTUP_S = 240                            # container start + model load + vLLM compile
MARGIN = 1.25
PRICE_PER_HOUR = 2.27                      # L40S + 4 CPU + 16 GiB


def load_targets(terminal_path=TERMINAL):
    """target -> {checkpoint (path on /vol), checksums of the adapter files}; original -> None."""
    jobs = json.loads(Path(terminal_path).read_text())["state"]["jobs"]
    specs = {"original": None}
    for run in TRAINED:
        result = jobs[run]["result"]
        if result.get("step") != 200 or result.get("partial") is not False:
            raise ValueError(f"{run}: terminal.json does not record a complete update-200 checkpoint")
        specs[run] = {"checkpoint": result["checkpoint"],
                      "checksums": {name: result["checksums"][name] for name in ADAPTER_FILES}}
    return specs


def verify_adapter(directory, spec):
    """Hash the adapter files in `directory`; raise if any differs from terminal.json."""
    from rl.checkpointing import file_hash
    for name, expected in spec["checksums"].items():
        actual = file_hash(Path(directory) / name)
        if actual != expected:
            raise ValueError(f"adapter checksum mismatch: {directory}/{name}: {actual} != {expected}")
    return spec["checksums"]["adapter_model.safetensors"]


def stages(target, k):
    """(name, dataset, k, sampled, output dir) in run order."""
    out = [("gsm8k-sampled", "gsm8k", k, True, f"{OUT_ROOT}/{target}/gsm8k-k{k}"),
           ("symbolic-greedy", "symbolic", 1, False, f"{OUT_ROOT}/{target}/symbolic")]
    if target == "original":
        out.append(("gsm8k-greedy", "gsm8k", 1, False, f"{OUT_ROOT}/{target}/gsm8k-greedy"))
    return out


def estimate_seconds(target, k):
    kind = "original" if target == "original" else "trained"
    sampled = MEASURED_256x64_S[kind] * (GSM8K_N / 256) * (k / 64)
    greedy = SYMBOLIC_GREEDY_S
    if target == "original":
        greedy += SYMBOLIC_GREEDY_S * GSM8K_N / SYMBOLIC_N   # GSM8K greedy, same per-question rate
    return {"sampled": sampled, "greedy": greedy, "startup": STARTUP_S,
            "total_with_margin": (sampled + greedy + STARTUP_S) * MARGIN}


def plan(targets, k, specs):
    if k not in (32, 64):
        raise ValueError("--k must be 32 or 64")
    unknown = set(targets) - set(TARGETS)
    if unknown or not targets or len(set(targets)) != len(targets):
        raise ValueError(f"bad targets: {targets}")
    rows = []
    for t in targets:
        est = estimate_seconds(t, k)
        rows.append({"target": t, "k": k,
                     "adapter": None if specs[t] is None else specs[t]["checkpoint"],
                     "adapter_sha256": None if specs[t] is None else specs[t]["checksums"]["adapter_model.safetensors"],
                     "outputs": [s[4] for s in stages(t, k)],
                     "seconds": est, "dollars": est["total_with_margin"] / 3600 * PRICE_PER_HOUR})
    return {"targets": rows, "k": k,
            "dollars_total": sum(r["dollars"] for r in rows),
            "dollars_without_margin": sum(r["dollars"] for r in rows) / MARGIN,
            "wall_clock_s_parallel": max(r["seconds"]["total_with_margin"] for r in rows),
            "wall_clock_s_parallel_no_margin": max(r["seconds"]["total_with_margin"] for r in rows) / MARGIN,
            "price_per_hour": PRICE_PER_HOUR}


def format_plan(p, max_parallel):
    lines = [f"Final evaluation plan: k={p['k']}, {len(p['targets'])} targets, up to {max_parallel} in parallel",
             f"GPU L40S, cpu=4, memory=16384 MiB, timeout {TIMEOUT_S // 3600} h, retries=0, "
             f"price ${p['price_per_hour']:.2f}/h"]
    for r in p["targets"]:
        s = r["seconds"]
        lines.append(f"- {r['target']}: adapter {r['adapter'] or '(none: base model)'}")
        if r["adapter_sha256"]:
            lines.append(f"    adapter sha256 {r['adapter_sha256']}")
        for o in r["outputs"]:
            lines.append(f"    -> {o}")
        lines.append(f"    est. sampled {s['sampled']/60:.1f} min + greedy {s['greedy']/60:.1f} min + startup "
                     f"{s['startup']/60:.0f} min, x{MARGIN} = {s['total_with_margin']/60:.1f} min, ${r['dollars']:.2f}")
    lines.append(f"Estimated total ${p['dollars_total']:.2f} (${p['dollars_without_margin']:.2f} without the "
                 f"{int((MARGIN-1)*100)}% margin); wall clock with all in parallel "
                 f"~{p['wall_clock_s_parallel_no_margin']/60:.0f} min (up to {p['wall_clock_s_parallel']/60:.0f} min with margin)")
    return "\n".join(lines)


def run_target(target, k, spec, time_limit_s, engine_factory, vol_root, manifest_paths, commit=None, now=time.time):
    """Body of the remote function, written so it can run locally with a fake engine.
    engine_factory(adapter_dir) -> engine. Paths in `spec` and the outputs are under vol_root;
    manifest_paths: {'gsm8k': path, 'symbolic': path}."""
    from rl.campaign import MODEL_REVISION
    from rl.evaluation_io import evaluate_manifest
    started = now()

    def guard():
        if now() - started >= time_limit_s:
            raise TimeoutError("final-eval per-call time limit")

    identity = {"base_revision": MODEL_REVISION, "target": target}
    adapter = None
    if target != "original":
        if spec is None:
            raise ValueError(f"{target}: no checkpoint spec")
        adapter = Path(vol_root) / Path(spec["checkpoint"]).relative_to(VOL)
        identity["adapter_sha256"] = verify_adapter(adapter, spec)
        adapter = str(adapter)
    engine = engine_factory(adapter)
    done = []
    try:
        for name, dataset, kk, sampled, out in stages(target, k):
            manifest = json.loads(Path(manifest_paths[dataset]).read_text())
            summary = evaluate_manifest(engine, manifest, Path(vol_root) / Path(out).relative_to(VOL), kk,
                                        sampled=sampled, model_identity=identity, commit=commit, guard=guard)
            done.append({"stage": name, "accuracy_strict": summary["accuracy_strict"], "passk": summary["passk"]})
    except TimeoutError as ex:
        return {"target": target, "complete": False, "stopped": str(ex), "done": done,
                "seconds": now() - started}
    return {"target": target, "complete": True, "done": done, "seconds": now() - started}


def create_app():
    from modal_app import campaign_image
    app = modal.App("lm-rl-final-eval", image=campaign_image(MANIFESTS), include_source=False)
    vol = modal.Volume.from_name("lm-rl")

    @app.function(gpu="L40S", cpu=4, memory=16384, timeout=TIMEOUT_S, retries=0, volumes={VOL: vol},
                  min_containers=0, buffer_containers=0, serialized=True, include_source=False)
    def evaluate_target(target: str, k: int, spec, time_limit_s: float):
        import sys as _sys
        os.chdir(CODE)
        if CODE not in _sys.path:
            _sys.path.insert(0, CODE)
        manifest_paths = {n: f"{CODE}/manifests/{n}.json" for n in ("gsm8k", "symbolic")}  # campaign_image puts them here
        vol.reload()

        def engine_factory(adapter_dir):
            from transformers import AutoTokenizer
            from rl.campaign import MODEL_REVISION, isolated_rng
            from rl.engine import VLLMEngine
            with isolated_rng():
                tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision=MODEL_REVISION)
                return VLLMEngine(None, tok, "Qwen/Qwen2.5-0.5B-Instruct", 512, "boxed",
                                  adapter_dir=adapter_dir, gpu_memory_utilization=0.5, revision=MODEL_REVISION)

        result = run_target(target, k, spec, time_limit_s, engine_factory, VOL, manifest_paths, commit=vol.commit)
        vol.commit()
        return result

    return app, evaluate_target


app = None
if modal.is_local():
    # Building the app object is lazy: nothing is built, deployed or started until `modal run`.
    app, evaluate_target = create_app()

    @app.local_entrypoint()
    def main(k: int, targets: str = ",".join(TARGETS), max_parallel: int = 7, dry_run: bool = False,
             time_limit_min: float = 0):
        chosen = [t.strip() for t in targets.split(",") if t.strip()]
        specs = load_targets()
        p = plan(chosen, k, specs)
        print(format_plan(p, max_parallel))
        if dry_run:
            print("DRY RUN: nothing started.")
            return
        limit = TIMEOUT_S - GUARD_MARGIN_S
        if time_limit_min > 0:
            limit = min(limit, time_limit_min * 60)
        print(f"per-call time limit: {limit / 60:.0f} min")
        pending = list(chosen)
        running, results = [], {}
        while pending or running:
            while pending and len(running) < max_parallel:
                t = pending.pop(0)
                running.append((t, evaluate_target.spawn(t, k, specs[t], limit)))
                print(f"spawned {t}")
            t, call = running.pop(0)
            try:
                results[t] = call.get()
            except Exception as ex:  # report and keep the other calls going
                results[t] = {"target": t, "complete": False, "error": repr(ex)}
            print(json.dumps(results[t]))
        print("SUMMARY", json.dumps({t: r.get("complete") for t, r in results.items()}))



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Plan the final evaluation (plain python: no Modal contact).")
    ap.add_argument("--k", type=int, required=True, choices=(32, 64))
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--max-parallel", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", required=True,
                    help="required: this form only prints the plan; the paid run is `modal run scripts/final_eval.py --k K`")
    a = ap.parse_args()
    print(format_plan(plan([t.strip() for t in a.targets.split(",") if t.strip()], a.k, load_targets()), a.max_parallel))
    print("DRY RUN: nothing started.")
