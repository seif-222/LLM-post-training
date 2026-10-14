"""
Smoke test of EVERY experiment in experiments.py (RUN_GPU=1 SMOKE_ALL=1, ~15 min on the laptop):
2 tiny steps each with the laptop settings, so a config bug shows up here, not on a paid GPU.
Each run is its own process (GPU memory fully freed). wandb is left out.
"""
import json
import os
import subprocess
import sys

import pytest

import experiments

pytestmark = pytest.mark.skipif(not (os.environ.get("RUN_GPU") == "1" and os.environ.get("SMOKE_ALL") == "1"),
                                reason="set RUN_GPU=1 SMOKE_ALL=1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 300 new tokens (not 128): at 128 almost every answer is cut off, every group scores 0 and the
# training step is skipped, so the smoke test would not test training at all.
TINY = ["--questions", "4", "--k", "4", "--max-new-tokens", "300", "--eval-every", "2", "--eval-n", "4",
        "--passk-n", "2", "--passk-k", "2", "--ckpt-every", "2", "--warmup", "1"]


@pytest.mark.parametrize("name", list(experiments.EXPERIMENTS))
def test_experiment_runs(name, tmp_path):
    _, argv = experiments.build(name, "laptop", lr="1e-4", steps=2, extra=[*TINY, "--out-root", str(tmp_path)])
    argv = [a for a in argv if a != "--wandb"]
    res = subprocess.run([sys.executable, os.path.join(ROOT, "train_grpo.py"), *argv], cwd=ROOT,
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-3000:]
    run = argv[argv.index("--run") + 1]
    rows = [json.loads(l) for l in open(tmp_path / run / "metrics.jsonl")]
    train = [r for r in rows if r["type"] == "train"]
    assert [r["step"] for r in train] == [1, 2]
    assert any("t_train" in r for r in train), "no training step ran (no mixed group in 2 steps)"
    assert os.path.exists(tmp_path / run / "adapter" / "adapter_model.safetensors")
