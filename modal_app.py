"""Explicitly authorized, deployed polarity campaign controller and workers.

Import without LM_RL_LAUNCH_MANIFEST is side-effect free. Preparation and dry-run:
    python scripts/run_campaign.py --prepare runs/preparation/001
A paid preflight run validates the remote runtime and the measured
budget/deadline manifest before this app is deployed or its controller submitted.
"""
import json
import os
import subprocess

import modal

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = "/root/lm_rl"                     # where the project code sits inside the container
VOL = "/vol"                             # the Modal disk
RUNS = f"{VOL}/runs/grpo"

# $ per hour: the GPU (modal.com/pricing, 2026-09) plus 4 CPU cores ($0.0473 each) and 16 GiB RAM
# ($0.008 per GiB) that every container here asks for.
GPU_PRICE = {"L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}
CPU_MEM_PRICE = 4 * 0.0473 + 16 * 0.008



def campaign_image(dataset_manifests):
    # Pinned to what the first working build resolved (2026-09-30); torch 2.13.0 and transformers
    # 5.18.0 come with vllm 0.30.0.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install("vllm==0.30.0", "transformers==5.18.0", "peft==0.21.1", "datasets==5.0.1",
                        "wandb==0.30.0", "pytest==9.1.1", "matplotlib==3.11.2")
        .env({"HF_HOME": "/root/hf", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TOKENIZERS_PARALLELISM": "false"})
        .run_commands("python -c \"from huggingface_hub import snapshot_download; from datasets import load_dataset; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775'); [load_dataset('openai/gsm8k', 'main', split=s, revision='740312add88f781978c0658806c59bc2815b9866') for s in ('train', 'test')]\"")
        .env({"PYTHONPATH": CODE})
        .add_local_dir(os.path.join(ROOT, "rl"), CODE + "/rl", ignore=["**/__pycache__"])
        .add_local_dir(os.path.join(ROOT, "scripts"), CODE + "/scripts", ignore=["**/__pycache__"])
        .add_local_file(os.path.join(ROOT, "train_grpo.py"), CODE + "/train_grpo.py")
        .add_local_file(os.path.join(ROOT, "experiments.py"), CODE + "/experiments.py")
        .add_local_file(os.path.join(ROOT, "modal_app.py"), CODE + "/modal_app.py")
        .add_local_dir(os.path.join(ROOT, "tests"), CODE + "/tests", ignore=["**/__pycache__"])
        .add_local_file(os.path.join(ROOT, "pytest.ini"), CODE + "/pytest.ini")

    )
    for name, source in dataset_manifests.items():
        if name not in ("gsm8k", "symbolic"):
            raise ValueError("unexpected dataset manifest")
        image = image.add_local_file(source, CODE + "/manifests/" + name + ".json")
    return image


def create_campaign_app(manifest):
    """Only explicit, resolved authorization can construct an image/deployment definition."""
    from rl.campaign import validate_authorization, validate_main_jobs, validate_training_jobs
    from rl.checkpointing import file_hash
    validate_authorization(manifest)
    if manifest["phase"] == "main":
        validate_main_jobs(manifest)
    elif manifest["phase"] == "training_only":
        validate_training_jobs(manifest)
    if not manifest.get("source_hashes") or not manifest.get("dataset_manifests") or not manifest.get("jobs"):
        raise ValueError("unresolved source hashes, datasets or measured job allocations")
    for relative, expected in manifest["source_hashes"].items():
        if os.path.isabs(relative) or ".." in relative.split("/") or file_hash(os.path.join(ROOT, relative)) != expected:
            raise ValueError("source hash mismatch")
    def download_assets():
        """Bake only the pinned Instruct model and GSM8K into the image."""
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")
        for split in ("train", "test"):
            load_dataset("openai/gsm8k", "main", split=split, revision="740312add88f781978c0658806c59bc2815b9866")


    image = campaign_image(manifest["dataset_manifests"])
    app = modal.App(manifest.get("deployment_name", "lm-rl-polarity"), image=image, include_source=False)
    vol = modal.Volume.from_name("lm-rl", create_if_missing=True)
    if manifest["phase"] == "training_only":
        from scripts.deadline_worker import install
        install(app, vol, manifest, CODE, VOL)
        return app
    common = dict(volumes={VOL: vol}, secrets=[modal.Secret.from_name("wandb-secret")], cpu=4, memory=16384)


    def _in_code_dir():
        import sys
        os.chdir(CODE)
        if CODE not in sys.path:
            sys.path.insert(0, CODE)


    @app.function(name="campaign_worker", gpu="L40S", timeout=4500, retries=0, max_containers=3, min_containers=0, buffer_containers=0, serialized=True, include_source=False, **common)
    def campaign_worker(job: dict, approved: dict):
        import time
        from pathlib import Path
        _in_code_dir()
        from rl.campaign import validate_authorization, worker_deadline, isolated_rng, hard_deadline
        validate_authorization(approved, now=job["dispatch_time"])
        claims = modal.Dict.from_name("lm-rl-polarity-worker-ownership", create_if_missing=True)
        claim_key = approved['campaign_id'] + "/" + job['logical_run'] + "/" + str(job['attempt'])
        if not claims.put(claim_key, {"call_id": modal.current_function_call_id(), "dispatch_time": job['dispatch_time']}, skip_if_exists=True):
            raise RuntimeError("logical-run attempt already owned; refusing duplicate worker")
        vol.reload()
        stop_at = worker_deadline(job["dispatch_time"], job["reserved_usd"], job["price_per_hour"],
                                  4500 if job["kind"] == "train" else 3600, approved["deadline_utc_epoch"])
        def guard():
            if time.time() >= stop_at - job["boundary_margin_seconds"]:
                raise TimeoutError("worker allocation/deadline guard")
        with hard_deadline(stop_at):
            guard()
            if job["kind"] == "train":
                import train_grpo
                argv = list(job["argv"]) + ["--out-root", f"{VOL}/campaigns/{approved['campaign_id']}/training"]
                run_dir = Path(VOL)/"campaigns"/approved['campaign_id']/"training"/job['logical_run']
                if job['attempt'] > 1:
                    if (run_dir/"ckpt/latest.json").exists():
                        argv.append("--resume")
                    elif run_dir.exists():
                        # This path is reachable only after a funded, confirmed-dead infrastructure
                        # failure before the first committed checkpoint. Retain the entire attempt.
                        import uuid
                        archived = run_dir.parent/"failed-initial-attempts"/(run_dir.name+"-"+uuid.uuid4().hex)
                        archived.parent.mkdir(parents=True, exist_ok=True)
                        run_dir.rename(archived)
                        vol.commit()
                train_grpo.main(argv, on_checkpoint=vol.commit, guard=guard)
            elif job["kind"] == "evaluation":
                from transformers import AutoTokenizer
                from rl.campaign import MODEL_REVISION
                from rl.engine import VLLMEngine
                from rl.evaluation_io import evaluate_manifest
                from rl.checkpointing import validate, file_hash
                target = job['logical_run'].removesuffix('-evaluation')
                adapter = None
                identity = {"base_revision": MODEL_REVISION, "target": target}
                if target != 'original':
                    training_root = Path(VOL)/"campaigns"/approved['campaign_id']/"training"/target
                    pointer = json.loads((training_root/"ckpt/latest.json").read_text())
                    generation = training_root/"ckpt"/pointer['generation']
                    completed = validate(generation)
                    if completed['step'] != 200:
                        raise ValueError("final evaluation requires committed update 200")
                    adapter = str(generation)
                    identity['adapter_sha256'] = file_hash(generation/"adapter_model.safetensors")
                with isolated_rng():
                    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision=MODEL_REVISION)
                    engine = VLLMEngine(None, tok, "Qwen/Qwen2.5-0.5B-Instruct", 512, "boxed",
                                        adapter_dir=adapter, gpu_memory_utilization=0.5,
                                        revision=MODEL_REVISION)
                out = Path(VOL)/"campaigns"/approved['campaign_id']/"evaluation"/job['logical_run']
                if target == 'original':
                    gsm_manifest = json.loads((Path(CODE)/"manifests/gsm8k.json").read_text())
                    evaluate_manifest(engine, gsm_manifest, out/"gsm8k-greedy", 1, sampled=False,
                                      model_identity=identity, commit=vol.commit, guard=guard)
                else:
                    import shutil
                    out.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(training_root/"evals/step200.json", out/"gsm8k-greedy.json")
                    vol.commit()
                for dataset, k, sampled in (("gsm8k", approved["k"], True), ("symbolic", 1, False)):
                    dataset_manifest = json.loads((Path(CODE)/"manifests"/(dataset+".json")).read_text())
                    evaluate_manifest(engine, dataset_manifest, out/dataset, k, sampled=sampled,
                                      model_identity=identity, commit=vol.commit, guard=guard)
            else:
                raise ValueError("unknown campaign job kind")
            vol.commit()
            return {"complete": True, "logical_run": job['logical_run'], "finished": time.time()}


    @app.function(name="campaign_controller", cpu=0.25, memory=256, timeout=21600, max_containers=1, min_containers=0, buffer_containers=0, serialized=True, include_source=False,
                  volumes={VOL: vol}, retries=0)
    def campaign_controller(approved: dict):
        import time
        from pathlib import Path
        _in_code_dir()
        from rl.campaign import validate_authorization, validate_main_jobs, Ledger, digest
        from rl.checkpointing import atomic_json
        validate_authorization(approved)
        if approved['phase'] == 'main':
            validate_main_jobs(approved)
        if approved['phase'] != 'main':
            raise ValueError("paid preflight requires its separately reviewed integration worker")
        owners = modal.Dict.from_name("lm-rl-polarity-ownership", create_if_missing=True)
        owner = approved['campaign_id']
        if not owners.put(owner, {"manifest_hash": digest(approved), "started": time.time()}, skip_if_exists=True):
            raise RuntimeError("campaign ownership exists; reconcile persisted call IDs before continuation")
        path = Path(VOL)/"campaigns"/owner/"controller.json"
        dashboard_url = app.get_dashboard_url()
        def persist(state):
            state["dashboard_url"] = dashboard_url
            atomic_json(path, state)
            vol.commit()
        ledger = Ledger(approved['budget_usd']-approved['controller_reservation_usd'], approved['deadline_utc_epoch'], persist, max_jobs=3)
        by_name = {job['logical_run']: job for job in approved['jobs']}
        if len(by_name) != len(approved['jobs']):
            raise ValueError("duplicate job identities")
        for wave in approved['waves']:
            calls = []
            for name in wave:
                job = by_name[name]
                reservation = ledger.reserve(name, job['allocation_usd'], job['seconds'], time.time())
                # Reservation is durable first. If spawn succeeds but persistence fails, ownership
                # remains uncertain and is never automatically submitted again.
                payload = {**job, **reservation}
                call = campaign_worker.with_options(timeout=int(job['seconds'])).spawn(payload, approved)
                ledger.dispatched(name, call.object_id)
                calls.append((name, call, payload))
            for name, call, payload in calls:
                while True:
                    try:
                        result = call.get(timeout=max(1, approved['deadline_utc_epoch']-time.time()))
                        if result.get('complete') is not True:
                            raise RuntimeError("worker returned incomplete")
                    except modal.exception.InternalFailure:
                        from modal.types import InputStatus
                        graph = call.get_call_graph()
                        def dead(node):
                            return node.status in (InputStatus.FAILURE, InputStatus.TERMINATED) and all(dead(child) for child in node.children)
                        if not graph or not all(dead(node) for node in graph):
                            raise RuntimeError("worker ownership remains uncertain; no retry")
                        ledger.finish(name, payload['reserved_usd'], False)
                        job = by_name[name]
                        reservation = ledger.reserve(name, job['allocation_usd'], job['seconds'], time.time(),
                                                     prior_dead=True, transient=True)
                        payload = {**job, **reservation}
                        call = campaign_worker.with_options(timeout=int(job['seconds'])).spawn(payload, approved)
                        ledger.dispatched(name, call.object_id)
                        continue
                    except Exception:
                        # Preserve uncertain ownership and the reserved budget. Reconciliation
                        # must establish terminal status before any future retry.
                        ledger.state['jobs'][name]['failure_observed'] = True
                        persist(ledger.state)
                        raise
                    ledger.state['jobs'][name]['billing_reconciled'] = False
                    ledger.state['jobs'][name]['accounting_basis'] = 'full reservation pending billing reconciliation'
                    ledger.finish(name, payload['reserved_usd'], True)
                    break
        return {"complete": True, "campaign_id": owner, "ledger": str(path), "dashboard_url": dashboard_url}

    return app

# Without authorization, importing this module constructs no image, volume or app.
_manifest_path = os.environ.get("LM_RL_LAUNCH_MANIFEST")
app = create_campaign_app(json.load(open(_manifest_path))) if _manifest_path else None
