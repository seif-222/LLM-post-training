"""
GPU test (RUN_GPU=1) of the trainer's vLLM-only path without vLLM: the gap logging and the
--tis-cap correction. An HF engine that also returns the sampling-time log-probs (rollout's
sample_logprobs) stands in for vLLM, so the trainer takes exactly the branch it takes with vLLM.
Here sampler and trainer are the same model, so the gap must be small (bf16 noise only).
"""
import json
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_GPU") != "1", reason="set RUN_GPU=1")


def test_gap_and_tis_path(tmp_path, monkeypatch):
    import train_grpo as tg
    from rl.engine import HFEngine
    from rl.eval import gen_config
    from rl.rollout import rollout

    class HFWithLogp(HFEngine):
        def sample(self, problems, k, temperature, requests=None):
            if requests is not None:
                return super().sample(problems, k, temperature, requests=requests)
            cfg = gen_config(self.model, True, temperature, self.max_new_tokens)
            return rollout(self.model, self.tok, problems, k, cfg, self.prompt_style, 64, sample_logprobs=True)

    monkeypatch.setattr(tg, "make_engine", lambda kind, model, tok, args, out:
                        HFWithLogp(model, tok, args.max_new_tokens, args.prompt))
    tg.main(["--run", "gap", "--out-root", str(tmp_path), "--overfit", "2", "--k", "4", "--max-new-tokens", "128",
             "--steps", "2", "--eval-every", "0", "--passk-n", "0", "--tis-cap", "2", "--lr", "1e-4", "--warmup", "1"])
    rows = [json.loads(l) for l in open(tmp_path / "gap" / "metrics.jsonl") if '"train"' in l]
    for r in rows:
        print({k: round(r[k], 5) for k in ("gap", "gap_abs", "gap_kl", "tis_capped_frac")})
        assert abs(r["gap"]) < 0.02 and r["gap_abs"] < 0.05 and 0 <= r["gap_kl"] < 0.01
        assert r["tis_capped_frac"] < 0.01
