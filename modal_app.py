"""
Modal launcher. Everything runs in a fresh image with vLLM; the laptop venv is untouched.
Run outputs live on the Modal disk (Volume "lm-rl") until downloaded into local runs/grpo/.

    modal run modal_app.py::test --gpu L4                       # test suite inside the image (CPU + GPU tests)
    modal run modal_app.py::bench                               # Stage B1: speed / cost table
    modal run modal_app.py::launch --group lr_sweep --gpu A100-40GB
    modal run modal_app.py::launch --group tier1 --gpu A100-40GB --lr 2e-5
    modal run modal_app.py::download                            # copy every run into runs/grpo/

A run that crashes or is interrupted is retried once and continues from its last checkpoint
(train_grpo --resume); the Volume is saved (committed) after every checkpoint.
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



def download_assets():
    """Bake both models and GSM8K into the image, so parallel containers never race to
    download them onto the shared disk."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    for m in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B"):
        snapshot_download(m)
    for split in ("train", "test"):
        load_dataset("openai/gsm8k", "main", split=split)


# Pinned to what the first working build resolved (2026-09-30); torch 2.13.0 and transformers
# 5.18.0 come with vllm 0.30.0.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0", "transformers==5.18.0", "peft==0.21.1", "datasets==5.0.1",
                    "wandb==0.30.0", "pytest==9.1.1", "matplotlib==3.11.2")
    .env({"HF_HOME": "/root/hf", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TOKENIZERS_PARALLELISM": "false"})
    .run_function(download_assets)
    .add_local_dir(ROOT, CODE, ignore=["runs", "notes", ".git", "**/__pycache__", "*.md", ".idea",
                                       ".delegate", ".venv", ".pytest_cache"])
)
app = modal.App("lm-rl", image=image)
vol = modal.Volume.from_name("lm-rl", create_if_missing=True)
common = dict(volumes={VOL: vol}, secrets=[modal.Secret.from_name("wandb-secret")], cpu=4, memory=16384)


def _in_code_dir():
    import sys
    os.chdir(CODE)
    if CODE not in sys.path:
        sys.path.insert(0, CODE)


@app.function(gpu="A100-40GB", timeout=6 * 3600, retries=modal.Retries(max_retries=1, initial_delay=10.0), **common)
def train(name: str, lr: str | None = None, steps: int | None = None, extra: list | None = None,
          out_root: str = RUNS, gpu_name: str = "A100-40GB"):
    """One experiment from experiments.py. Resumes automatically if the run already has metrics."""
    _in_code_dir()
    import experiments
    import train_grpo
    vol.reload()
    _, argv = experiments.build(name, "modal", lr=lr, steps=steps, extra=extra or [])
    argv += ["--out-root", out_root, "--price-per-hour", str(GPU_PRICE[gpu_name] + CPU_MEM_PRICE)]
    run = train_grpo.parse_args(argv).run                 # extra may rename the run (bench does)
    if os.path.exists(os.path.join(out_root, run, "metrics.jsonl")):
        argv.append("--resume")
    print("train_grpo", " ".join(argv), flush=True)
    train_grpo.main(argv, on_checkpoint=vol.commit)
    vol.commit()
    return run


@app.function(gpu="L4", timeout=3600, **common)
def run_tests(path: str = ""):
    """The test suite inside the image: CPU tests + GPU tests (vLLM agreement, top-20 check).
    path = one test file only (e.g. tests/test_gpu_vllm.py)."""
    _in_code_dir()
    code = subprocess.run(["python", "-m", "pytest", "-q", "-rsP", *([path] if path else [])],
                          env={**os.environ, "RUN_GPU": "1"}).returncode
    subprocess.run(["pip", "freeze"], check=False)       # record the versions to pin
    return code


@app.function(gpu="A100-40GB", timeout=3600, **common)
def bench_packing(n_adapters: int = 4, questions: int = 24, k: int = 8):
    """Packing test (plan B1): generate one step's round 0 (24 questions x 8 = 192 answers) for
    1 adapter, then for n adapters in ONE vLLM batch (each request tagged with its own adapter).
    If n adapters take much less than n times as long, packing pays off on the generate part."""
    _in_code_dir()
    import time
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from rl.data import build_prompt, load_gsm8k
    from rl.engine import VLLMEngine

    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda(),
                           LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0))
    dirs = []
    for i in range(n_adapters):                                # n different, non-zero adapters
        torch.manual_seed(i)
        for n_, p in model.named_parameters():
            if "lora_B" in n_:
                p.data.normal_(0, 1e-3)
        dirs.append(f"/tmp/adapter{i}")
        model.save_pretrained(dirs[-1])
    eng = VLLMEngine(model, tok, name, 512, adapter_dir="/tmp/adapter_sync", max_loras=n_adapters,
                     gpu_memory_utilization=0.5)
    problems = load_gsm8k("train")[:questions]
    eng.sample(problems[:4], k, 1.0)                           # warm-up (CUDA graphs, first LoRA load)
    t0 = time.time()
    eng.sample(problems, k, 1.0)
    t_one = time.time() - t0
    prompts = [TokensPrompt(prompt_token_ids=tok(build_prompt(tok, p["question"], "boxed"))["input_ids"])
               for p in problems]
    loras = [LoRARequest(f"a{i}", 100 + i, d) for i, d in enumerate(dirs)]
    eng.llm.generate(prompts[:2] * n_adapters, eng._params(k, 1.0, True),
                     lora_request=[l for l in loras for _ in range(2)], use_tqdm=False)   # load all adapters
    t0 = time.time()
    eng.llm.generate(prompts * n_adapters, eng._params(k, 1.0, True),
                     lora_request=[l for l in loras for _ in prompts], use_tqdm=False)
    t_n = time.time() - t0
    return {"n_adapters": n_adapters, "answers_per_adapter": questions * k, "gen_1_adapter_s": round(t_one, 1),
            f"gen_{n_adapters}_adapters_s": round(t_n, 1), "per_adapter_s": round(t_n / n_adapters, 1)}


@app.function(gpu="L40S", timeout=3600, **common)
def evaluate(argv: list, out: str):
    """python -m rl.eval <argv> inside the image; results go to the Modal disk (out = a path there)."""
    _in_code_dir()
    vol.reload()
    code = subprocess.run(["python", "-m", "rl.eval", *argv, "--engine", "vllm", "--out", out]).returncode
    vol.commit()
    return code


@app.function(timeout=600, **common)
def read_metrics(out_root: str = RUNS):
    """{run: [metrics rows]} for every run on the Volume (for the bench report)."""
    vol.reload()
    out = {}
    if os.path.isdir(out_root):
        for run in sorted(os.listdir(out_root)):
            p = os.path.join(out_root, run, "metrics.jsonl")
            if os.path.exists(p):
                out[run] = [json.loads(l) for l in open(p)]
    return out


@app.local_entrypoint()
def launch(group: str, gpu: str = "A100-40GB", lr: str = "", steps: int = 0):
    """Start every experiment of a group in parallel, each in its own container."""
    import experiments
    names = experiments.GROUPS.get(group, [group])
    fn = train.with_options(gpu=gpu)
    calls = [fn.spawn(n, lr or None, steps or None, None, RUNS, gpu) for n in names]
    print(f"launched {len(calls)} runs on {gpu}: {', '.join(names)}")
    for c in calls:
        print("finished:", c.get())


@app.local_entrypoint()
def test(gpu: str = "L4", path: str = ""):
    print("exit code", run_tests.with_options(gpu=gpu).remote(path))


@app.local_entrypoint()
def bench():
    """Stage B1. 5 real steps per setup (the first is warm-up), evals off, then the table:
    seconds per step, split into generate / train, and $ per 1,000 training questions."""
    root = f"{VOL}/runs/bench"
    setups = [  # (run name, gpu, extra flags, questions per step)
        ("vllm_L4", "L4", ["--engine", "vllm"], 16),
        ("vllm_L40S", "L40S", ["--engine", "vllm"], 16),
        ("vllm_A100", "A100-40GB", ["--engine", "vllm"], 16),
        ("hf_A100", "A100-40GB", ["--engine", "hf"], 16),
        ("vllm_A100_64x8", "A100-40GB", ["--engine", "vllm", "--questions", "64"], 64),
    ]
    common_flags = ["--eval-every", "0", "--passk-n", "0", "--ckpt-every", "0"]
    calls = [(run, gpu, q, train.with_options(gpu=gpu).spawn("lr_2e-5", "2e-5", 5,
                                                              [*flags, *common_flags, "--run", f"bench_{run}"],
                                                              root, gpu))
             for run, gpu, flags, q in setups]
    pack = bench_packing.spawn()
    for c in calls:
        c[3].get()
    metrics = read_metrics.remote(root)
    print(f"{'setup':16} {'s/step':>7} {'generate':>9} {'train':>6} {'$/h':>5} {'$ per 1k questions':>19}")
    for run, gpu, q, _ in calls:
        rows = [r for r in metrics.get(f"bench_{run}", []) if r["type"] == "train"][1:]   # drop warm-up
        if not rows:
            print(f"{run:16} (no data)")
            continue
        s = sum(r["t_step"] for r in rows) / len(rows)
        g = sum(r["t_gen"] for r in rows) / len(rows)
        t = sum(r.get("t_train", 0) for r in rows) / len(rows)
        price = GPU_PRICE[gpu] + CPU_MEM_PRICE
        print(f"{run:16} {s:7.1f} {g:9.1f} {t:6.1f} {price:5.2f} {price * s / 3600 * 1000 / q:19.3f}")
    print("packing:", pack.get())


@app.local_entrypoint()
def passk(k: int = 64, k_n: int = 256, runs: str = "lr_1e-4_modal"):
    """pass@1..k on k_n test questions: the original model plus each run's final adapter, in parallel."""
    jobs = [([], f"{VOL}/runs/eval/original_k{k}")]
    jobs += [(["--adapter", f"{RUNS}/{r}/adapter"], f"{VOL}/runs/eval/{r}_k{k}") for r in runs.split(",")]
    calls = [evaluate.spawn([*a, "--n", "0", "--k", str(k), "--k-n", str(k_n)], out) for a, out in jobs]
    for (a, out), c in zip(jobs, calls):
        print(out, "exit", c.get())


@app.local_entrypoint()
def download(run: str = ""):
    """Copy runs from the Modal disk into local runs/grpo/ (the master copy). Skips files that
    already exist locally with the same size."""
    from modal.volume import FileEntryType
    src = f"runs/grpo/{run}" if run else "runs/grpo"
    n = 0
    for entry in vol.listdir(src, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        local = os.path.join(ROOT, entry.path)
        if os.path.exists(local) and os.path.getsize(local) == entry.size:
            continue
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as f:
            vol.read_file_into_fileobj(entry.path, f)
        n += 1
    print(f"downloaded {n} files into runs/grpo/")
