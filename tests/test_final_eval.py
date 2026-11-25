"""scripts/final_eval.py and final_eval_links.py without a GPU or any Modal call."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TERMINAL = ROOT / "runs/continuation/20261101/terminal.json"
GSM8K = ROOT / "runs/preparation/001/gsm8k-test-manifest.json"
SYMBOLIC = ROOT / "runs/preparation/001/symbolic-test-manifest.json"
pytestmark = pytest.mark.skipif(not (TERMINAL.exists() and GSM8K.exists() and SYMBOLIC.exists()),
                                reason="local campaign records not present")


@pytest.fixture(scope="module")
def fe():
    import scripts.final_eval as fe
    return fe


class FakeSampler:
    """Every sample is '\\boxed{1}' (two tokens), as in tests/test_polarity.py."""
    def __init__(self):
        self.calls = []

    def sample(self, questions, k, temperature, requests=None):
        from rl.engine import pack_completions

        class Tok:
            pad_token_id = 9

            def batch_decode(self, ids, **kwargs):
                return ["\\boxed{1}"] * len(ids)
        self.calls.append(([q["id"] for q in questions], k, temperature))
        n = len(questions) * k
        return pack_completions([[1] for _ in questions], [[3, 9]] * n, [[0., 0.]] * n, [True] * n, k, Tok(), [9])

    def greedy(self, questions):
        raise AssertionError("not reached in this test")


def test_targets_and_plan_from_real_terminal(fe):
    specs = fe.load_targets()
    assert list(specs) == fe.TARGETS and specs["original"] is None
    for run in fe.TRAINED:
        assert specs[run]["checkpoint"].startswith(f"/vol/continuations/resume-20261101-0006-r2/training/{run}/ckpt/update-0200-")
        assert set(specs[run]["checksums"]) == {"adapter_model.safetensors", "adapter_config.json"}
    for k in (64, 32):
        p = fe.plan(fe.TARGETS, k, specs)
        assert [r["target"] for r in p["targets"]] == fe.TARGETS
        orig = p["targets"][0]
        assert orig["outputs"] == [f"/vol/final-eval/20261123/original/gsm8k-k{k}",
                                   "/vol/final-eval/20261123/original/symbolic",
                                   "/vol/final-eval/20261123/original/gsm8k-greedy"]
        assert all(len(r["outputs"]) == 2 for r in p["targets"][1:])
        text = fe.format_plan(p, 7)
        assert "full_s0" in text and "Estimated total" in text
    assert 4 < fe.plan(fe.TARGETS, 64, specs)["dollars_total"] < 15
    with pytest.raises(ValueError):
        fe.plan(fe.TARGETS, 16, specs)
    with pytest.raises(ValueError):
        fe.plan(["original", "bogus"], 64, specs)


def make_checkpoint(tmp_path, fe, tamper=False):
    """A fake /vol with one checkpoint whose adapter hashes are what spec records."""
    spec = fe.load_targets()["full_s0"]
    vol_root = tmp_path / "vol"
    ckpt = vol_root / Path(spec["checkpoint"]).relative_to("/vol")
    ckpt.mkdir(parents=True)
    checksums = {}
    for name in fe.ADAPTER_FILES:
        (ckpt / name).write_bytes(name.encode())
        checksums[name] = hashlib.sha256(name.encode()).hexdigest()
    if tamper:
        (ckpt / "adapter_model.safetensors").write_bytes(b"different weights")
    return vol_root, {"checkpoint": spec["checkpoint"], "checksums": checksums}


def test_one_real_gsm8k_chunk_then_clean_stop(tmp_path, fe):
    vol_root, spec = make_checkpoint(tmp_path, fe)
    engine, seen = FakeSampler(), []
    clock = iter([0.0, 0.0, 1e9, 1e9, 1e9])   # start, guard before chunk 0, guard before chunk 1 -> stop

    def factory(adapter_dir):
        seen.append(adapter_dir)
        return engine
    result = fe.run_target("full_s0", 32, spec, 100.0, factory, vol_root,
                           {"gsm8k": str(GSM8K), "symbolic": str(SYMBOLIC)}, now=lambda: next(clock))
    assert result["complete"] is False and "time limit" in result["stopped"] and result["done"] == []
    assert seen == [str(vol_root / Path(spec["checkpoint"]).relative_to("/vol"))]
    assert len(engine.calls) == 1 and engine.calls[0][0] == list(range(32)) and engine.calls[0][1:] == (32, 1.0)
    out = vol_root / "final-eval/20261123/full_s0/gsm8k-k32"
    config = json.loads((out / "config.json").read_text())
    assert config["k"] == 32 and config["sampled"] is True
    assert config["model"]["adapter_sha256"] == spec["checksums"]["adapter_model.safetensors"]
    records = json.loads((out / "chunks/0000.json").read_text())
    assert [r["id"] for r in records] == list(range(32))
    assert all(len(r["samples"]) == 32 for r in records)
    gold = {q["id"]: q["answer"] for q in json.loads(GSM8K.read_text())["questions"][:32]}
    assert all(r["n_correct"] == (32 if gold[r["id"]] == 1.0 else 0) for r in records)
    assert not (out / "chunks/0001.json").exists() and not (out / "summary.json").exists()


def test_checksum_mismatch_is_refused(tmp_path, fe):
    vol_root, spec = make_checkpoint(tmp_path, fe, tamper=True)
    started = []
    with pytest.raises(ValueError, match="checksum mismatch"):
        fe.run_target("full_s0", 64, spec, 100.0, lambda a: started.append(a), vol_root,
                      {"gsm8k": str(GSM8K), "symbolic": str(SYMBOLIC)})
    assert started == []          # no engine built, nothing evaluated
    assert not (vol_root / "final-eval").exists()


def test_links_and_matched_format(tmp_path):
    import scripts.final_eval_links as fl
    training = tmp_path / "training"
    for t in fl.TARGETS:
        dirs = [f"gsm8k-k64", "symbolic"] + (["gsm8k-greedy"] if t == "original" else [])
        for d in dirs:
            (tmp_path / "eval" / t / d).mkdir(parents=True)
            (tmp_path / "eval" / t / d / "summary.json").write_text(json.dumps({"complete": True}))
        if t != "original":
            (training / t / "evals").mkdir(parents=True)
            (training / t / "evals/step200.json").write_text(json.dumps(
                {"records": [{"id": i, "correct_strict": True} for i in range(1319)]}))
    out, problems = fl.links(tmp_path / "eval", 64, training)
    assert problems == []
    matched = json.loads((tmp_path / "eval/matched-gsm8k.json").read_text())
    assert matched["original"].endswith("original/gsm8k-greedy")
    assert matched["full_s0"].endswith("full_s0/evals/step200.json")
    assert set(json.loads((tmp_path / "eval/campaign-symbolic.json").read_text())) == set(fl.TARGETS)
    # The downloaded step200 files, if present, are in the format compare.py's matched loader reads.
    real = ROOT / "runs/final/training/full_s0/evals/step200.json"
    if real.exists():
        recs = json.loads(real.read_text())["records"]
        assert len(recs) == 1319 and all(isinstance(r["correct_strict"], bool) for r in recs)
