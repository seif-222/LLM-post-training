"""
GPU check: the log-probs we train on must be the log-probs we sampled from.

Every GRPO formula assumes they match. We also check the position-id question under left
padding: for a RoPE model like Qwen, arange positions turn out to be harmless, because they
shift every real token of a row by the same constant, and RoPE only sees distances.
Runs in float32, so any real bug stands far above rounding noise.

    RUN_GPU=1 python -m pytest tests/test_gpu_logprobs.py -s
"""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get("RUN_GPU")),
                                reason="needs CUDA and RUN_GPU=1")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def setup():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl.data import load_gsm8k
    from rl.eval import gen_config

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda()
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0))
    torch.manual_seed(0)
    for n, p in model.named_parameters():          # LoRA B starts at 0; make the adapter matter
        if "lora_B" in n:
            p.data.normal_(0, 0.02)
    model.eval()
    # Questions of very different lengths -> heavy left padding on the short ones.
    train = load_gsm8k("train")
    problems = sorted(train[:40], key=lambda p: len(p["question"]))
    problems = [problems[0], problems[-1], problems[20]]
    cfg = gen_config(model, True, 1.0, 64)
    return model, tok, problems, cfg


def _run(setup):
    from rl.logprobs import token_logprobs
    from rl.rollout import rollout
    model, tok, problems, cfg = setup
    torch.manual_seed(1)
    ro = rollout(model, tok, problems, 2, cfg, sample_logprobs=True)
    P = ro["prompt_len"]
    batch = {k: ro[k].cuda() for k in ("input_ids", "attention_mask", "completion_mask")}
    with torch.no_grad():
        lp = token_logprobs(model, **batch, prompt_len=P)["logp"].cpu()
    return ro, lp, batch, P


def test_train_logprobs_match_sampling(setup):
    ro, lp, _, P = _run(setup)
    m = ro["completion_mask"][:, P:].bool()
    diff = (lp - ro["sample_logp"]).abs()[m]
    print(f"\n[fp32 + LoRA] tokens={m.sum().item()}  max|diff|={diff.max():.2e}  mean={diff.mean():.2e}")
    assert diff.max() < 2e-3


def test_arange_positions_only_shift_rope(setup):
    """The same forward pass with positions = column index. A left-padded row's real tokens are
    all shifted by (#pads). RoPE attention depends on q/k distance only, so nothing changes."""
    from rl.logprobs import _base
    model = setup[0]
    ro, _, batch, P = _run(setup)
    base = _base(model)
    with torch.no_grad():
        L = batch["input_ids"].shape[1]
        pos = torch.arange(L, device="cuda").expand_as(batch["input_ids"])
        h = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       position_ids=pos).last_hidden_state
        logits = base.lm_head(h[:, P - 1:-1]).float()
        lp_bad = logits.log_softmax(-1).gather(-1, batch["input_ids"][:, P:, None]).squeeze(-1).cpu()
    m = ro["completion_mask"][:, P:].bool()
    padded_rows = (batch["attention_mask"][:, :P] == 0).any(1).cpu()
    diff = (lp_bad - ro["sample_logp"]).abs()
    bad = diff[padded_rows.unsqueeze(1) & m]
    good = diff[~padded_rows.unsqueeze(1) & m]
    print(f"\n[arange] padded rows mean|diff|={bad.mean():.2e}   unpadded rows mean|diff|={good.mean():.2e}")
    assert bad.max() < 2e-3 and good.max() < 2e-3


def test_disable_adapter_is_base_model(setup):
    """The KL reference is 'LoRA switched off': it must equal a model without LoRA."""
    from rl.logprobs import token_logprobs
    model = setup[0]
    ro, lp_lora, batch, P = _run(setup)
    with torch.no_grad(), model.disable_adapter():
        lp_ref = token_logprobs(model, **batch, prompt_len=P)["logp"].cpu()
    from transformers import AutoModelForCausalLM
    plain = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        lp_plain = token_logprobs(plain, **batch, prompt_len=P)["logp"].cpu()
    m = ro["completion_mask"][:, P:].bool()
    assert (lp_ref - lp_plain).abs()[m].max() < 1e-4
    assert (lp_ref - lp_lora).abs()[m].mean() > 1e-3          # the adapter really was active
    del plain
