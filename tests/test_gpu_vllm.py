"""
GPU tests of the vLLM engine (RUN_GPU=1 and vllm installed; they run inside the Modal image,
`modal run modal_app.py::test`).

1. Agreement: the trainer's log p of each token vLLM sampled is close to vLLM's own number
   (the training-inference gap). Measured with a non-zero LoRA, so the adapter hand-over is tested too.
2. No hidden top-k: at T=1 a few sampled tokens must fall OUTSIDE the trainer's top 20. If vLLM
   had silently applied Qwen's top_k=20 (its default generation_config), there would be none.
3. The adapter really reaches vLLM: a big random adapter changes vLLM's greedy answers.
"""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("RUN_GPU") != "1", reason="set RUN_GPU=1")
vllm = pytest.importorskip("vllm")

NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl.engine import VLLMEngine
    tok = AutoTokenizer.from_pretrained(NAME)
    tok.padding_side = "left"
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).cuda(),
                           LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0))
    torch.manual_seed(0)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 1e-3)                  # small but non-zero: the policy differs from the base
    eng = VLLMEngine(model, tok, NAME, 384, adapter_dir=str(tmp_path_factory.mktemp("adapter")),
                     gpu_memory_utilization=0.35)
    return model.eval(), tok, eng


def test_logprob_agreement_and_no_hidden_topk(setup):
    from rl.data import load_gsm8k
    from rl.logprobs import _base
    from rl.rollout import position_ids_from_mask
    from rl.logprobs import token_logprobs
    model, tok, eng = setup
    problems = load_gsm8k("train")[:16]
    ro = eng.sample(problems, 4, 1.0)
    P = ro["prompt_len"]
    dev = "cuda"
    batch = {k: ro[k].to(dev) for k in ("input_ids", "attention_mask", "completion_mask")}
    cm = batch["completion_mask"][:, P:].bool()
    with torch.no_grad():
        logp = torch.cat([token_logprobs(model, **{k: v[i:i + 8] for k, v in batch.items()}, prompt_len=P,
                                         top_p=None)["logp"] for i in range(0, len(ro["texts"]), 8)])
    d = (logp - ro["sample_logp"].to(dev))[cm]
    print(f"tokens {cm.sum().item()} | mean gap {d.mean().item():+.5f} | mean |gap| {d.abs().mean().item():.5f}")
    # Yardstick: HF generate vs the trainer, SAME model in bf16, already differ by ~0.011 per token
    # on average (tests/test_gpu_gap.py). vLLM should be within a few times that.
    assert d.abs().mean().item() < 0.05
    assert abs(d.mean().item()) < 0.01

    # Rank of every sampled token under the trainer's own distribution.
    base = _base(model)
    outside = 0
    with torch.no_grad():
        for i in range(0, len(ro["texts"]), 8):
            ids, am = batch["input_ids"][i:i + 8], batch["attention_mask"][i:i + 8]
            h = base.model(input_ids=ids, attention_mask=am, position_ids=position_ids_from_mask(am)).last_hidden_state
            h = h[:, P - 1:-1][cm[i:i + 8]]
            tgt = ids[:, P:][cm[i:i + 8]]
            logits = (h @ base.lm_head.weight.T).float()
            rank = (logits > logits.gather(-1, tgt[:, None])).sum(-1)
            outside += (rank >= 20).sum().item()
    print(f"sampled tokens outside the top 20: {outside}")
    assert outside > 0, "no sampled token outside the top 20: a hidden top-k is on"


def _gap(model, eng, problems):
    """Sample with vLLM, re-score the same tokens with the trainer -> mean |log-prob gap| per token."""
    from rl.logprobs import token_logprobs
    ro = eng.sample(problems, 2, 1.0)
    P = ro["prompt_len"]
    batch = {k: ro[k].cuda() for k in ("input_ids", "attention_mask", "completion_mask")}
    cm = batch["completion_mask"][:, P:].bool()
    with torch.no_grad():
        logp = torch.cat([token_logprobs(model, **{k: v[i:i + 8] for k, v in batch.items()}, prompt_len=P,
                                         top_p=None)["logp"] for i in range(0, len(ro["texts"]), 8)])
    return (logp - ro["sample_logp"].cuda())[cm].abs().mean().item()


def test_adapter_reaches_vllm(setup):
    """After every sync, vLLM must sample from the trainer's CURRENT weights: the trainer re-scores
    vLLM's tokens with a small gap. A stale adapter in vLLM (old weights) would give a large gap,
    because the change below moves the model a lot."""
    from rl.data import load_gsm8k
    model, tok, eng = setup
    problems = load_gsm8k("test")[:8]
    g0 = _gap(model, eng, problems)
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.05)                  # a big random change
    eng.sync(1)
    g1 = _gap(model, eng, problems)
    for n, p in model.named_parameters():
        if n in saved:
            p.data.copy_(saved[n])
    g_stale = _gap(model, eng, problems)             # vLLM still has step 1's weights: must disagree
    eng.sync(2)
    g2 = _gap(model, eng, problems)
    print(f"gap: start {g0:.4f} | after change {g1:.4f} | stale on purpose {g_stale:.4f} | restored {g2:.4f}")
    assert g_stale > 0.2, "the change is too small to tell a stale adapter apart"
    assert max(g0, g2) < 0.05
    # The big random change scrambles the model: bf16 rounding grows with it (0.18 measured on
    # Modal, 2026-09-30, vs 4.3 when stale). Fresh weights = far below the stale gap.
    assert g1 < g_stale / 5
