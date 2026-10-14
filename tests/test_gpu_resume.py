"""
GPU test of checkpoint + resume (RUN_GPU=1, ~1.5 min on the laptop).

Why not "4 steps straight == 2 + resume + 2": two fresh runs with the same seed already differ
after one step. The backward pass on the GPU is not bit-exact (step-1 grad norm 0.531544 vs
0.531530 in two identical runs), the weights differ by a hair, and sampling at T=1 amplifies
that: at step 2, 3 of 8 answers were different. So we test what resume promises instead:
two resumes from the SAME checkpoint must pick the same questions, sample the same answers
(generation runs before any backward) and get the same loss; only numbers that come out of
the backward pass (grad norm) may differ, within a small tolerance.
"""
import gzip
import json
import os
import shutil

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("RUN_GPU") != "1", reason="set RUN_GPU=1")


def test_two_resumes_from_one_checkpoint_agree(tmp_path):
    import train_grpo as tg
    common = ["--out-root", str(tmp_path), "--overfit", "2", "--k", "4", "--max-new-tokens", "160",
              "--eval-every", "0", "--passk-n", "0", "--ckpt-every", "2", "--lr", "1e-4", "--warmup", "2"]
    tg.main(["--run", "a", "--steps", "2", *common])
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    for run in ("a", "b"):
        torch.cuda.empty_cache()
        tg.main(["--run", run, "--steps", "3", "--resume", *common])

    def step3(run):
        rows = [json.loads(l) for l in open(tmp_path / run / "metrics.jsonl")]
        with gzip.open(tmp_path / run / "samples.jsonl.gz", "rt") as f:
            samples = [json.loads(l) for l in f]
        assert [r["step"] for r in rows if r["type"] == "train"] == [1, 2, 3]   # nothing duplicated
        return next(r for r in rows if r["step"] == 3), [s for s in samples if s["step"] == 3]

    (ma, sa), (mb, sb) = step3("a"), step3("b")
    assert [s["idx"] for s in sa] == [s["idx"] for s in sb]          # same questions (stream restored)
    assert [s["text"] for s in sa] == [s["text"] for s in sb]        # same answers (weights + RNG restored)
    assert ma["lr"] == mb["lr"] and ma["acc"] == mb["acc"]
    assert ma["loss"] == pytest.approx(mb["loss"], abs=1e-6)
    assert ma["grad_norm"] == pytest.approx(mb["grad_norm"], rel=1e-3)
