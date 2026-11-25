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
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("RUN_GPU") != "1", reason="set RUN_GPU=1")
vllm = pytest.importorskip("vllm")

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
from rl.campaign import MODEL_REVISION, DATA_REVISION


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl.engine import VLLMEngine
    tok = AutoTokenizer.from_pretrained(NAME, revision=MODEL_REVISION)
    tok.padding_side = "left"
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(NAME, revision=MODEL_REVISION, dtype=torch.bfloat16).cuda(),
                           LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0))
    torch.manual_seed(0)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 1e-3)                  # small but non-zero: the policy differs from the base
    eng = VLLMEngine(model, tok, NAME, 384, adapter_dir=str(tmp_path_factory.mktemp("adapter")),
                     gpu_memory_utilization=0.35, revision=MODEL_REVISION)
    return model.eval(), tok, eng


def test_logprob_agreement_and_no_hidden_topk(setup):
    from rl.data import load_gsm8k
    from rl.logprobs import _base
    from rl.rollout import position_ids_from_mask
    from rl.logprobs import token_logprobs
    model, tok, eng = setup
    problems = load_gsm8k("train", revision=DATA_REVISION)[:16]
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


def test_adapter_reaches_vllm(setup, tmp_path):
    """Causal cache/propagation test on FIXED tokens, then normal optimizer updates.

    The old g1 < g_stale/5 compared different sampled texts and different trainer
    adapters. A huge random adapter amplifies cross-engine bf16 error, so that
    ratio did not isolate whether sync loaded current weights. Here current sync
    must distinguish an independently saved/reloaded identical adapter from
    deliberately stale weights on the SAME tokens. LoRA CUDA reductions use
    atomics, so identical inputs/weights need not be bit-exact; record an
    identical-request repeat control. Stale discrepancy must exceed both the
    reload discrepancy and repeat noise by the original factor of five, and0.2.
    Cross-engine normal-weight tolerances remain 0.05 absolute / 0.01 signed,
    including after two actual optimizer steps. Large-adapter HF disagreement
    is recorded separately, never used as evidence that stale weights loaded.
    """
    import json, shutil, hashlib
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from rl.data import load_gsm8k
    from rl.campaign import training_requests, finite
    from rl.logprobs import token_logprobs
    from rl.checkpointing import file_hash
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    model,tok,eng=setup
    problems=load_gsm8k('test',revision=DATA_REVISION)[:8]
    ro=eng.sample(problems,1,1.0,requests=training_requests(1234,0,0,[p['id'] for p in problems]))
    P=ro['prompt_len']
    batch={k:ro[k][:,:P+64].cuda() for k in ('input_ids','attention_mask','completion_mask')}
    mask=batch['completion_mask'][:,P:].bool()
    rows=[]; starts=[]
    for ids,am in zip(batch['input_ids'],batch['attention_mask']):
        rows.append(ids[am.bool()].cpu().tolist())
        starts.append(int(am[:P].sum()))
    def hf_scores(grad=False):
        pieces=[]
        with torch.set_grad_enabled(grad):
            for i in range(0,len(rows),4):
                out=token_logprobs(model,**{k:v[i:i+4] for k,v in batch.items()},prompt_len=P,top_p=None)['logp']
                pieces.append(out[mask[i:i+4]])
        return torch.cat(pieces)
    def score(request):
        outputs=eng.llm.generate([TokensPrompt(prompt_token_ids=r) for r in rows],
                                SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,seed=1234),
                                lora_request=request,use_tqdm=False)
        values=[]
        for ids,start,out in zip(rows,starts,outputs):
            assert len(out.prompt_logprobs)==len(ids)
            values.extend(out.prompt_logprobs[i][ids[i]].logprob for i in range(start,len(ids)))
        return torch.tensor(values,device='cuda')
    def saved_identity(directory):
        disk=load_file(str(directory/'adapter_model.safetensors'))
        current=get_peft_model_state_dict(model)
        assert set(disk)==set(current)
        for key in disk: assert torch.equal(disk[key],current[key].detach().cpu()),key
        return file_hash(directory/'adapter_model.safetensors')
    saved={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    old_dir=tmp_path/'independent-original'; model.save_pretrained(old_dir)
    original_hash=saved_identity(old_dir)
    stale_request=LoRARequest('deliberately-stale',10001,str(old_dir))
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(0)
        for n,p in model.named_parameters():
            if 'lora_B' in n:p.data.normal_(0,0.05)
    eng.sync(1)
    current_hash=saved_identity(Path(eng.lora.lora_path))
    assert current_hash!=original_hash
    independent=tmp_path/'independent-large'; model.save_pretrained(independent)
    assert saved_identity(independent)==current_hash
    fresh=score(eng.lora)
    repeated=score(eng.lora)
    independent_scores=score(LoRARequest('independent-large',10002,str(independent)))
    stale=score(stale_request)
    same=(fresh-independent_scores).abs()
    stale_gap=(fresh-stale).abs().mean().item()
    hf_large=(hf_scores()-fresh).abs().mean().item()
    evidence={'fixed_token_sha256':hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
              'tokens':fresh.numel(),'original_adapter_sha256':original_hash,'large_adapter_sha256':current_hash,
              'current_vs_independent_max':same.max().item(),'current_vs_independent_mean':same.mean().item(),
              'identical_request_repeat_max':(fresh-repeated).abs().max().item(),
              'identical_request_repeat_mean':(fresh-repeated).abs().mean().item(),
              'current_vs_stale_mean':stale_gap,'large_hf_vs_vllm_mean':hf_large,'optimizer_updates':[]}
    print('FIXED TOKEN DIAGNOSTIC',json.dumps(evidence),flush=True)
    # Numerical reproducibility is measured, not assumed: same weights/tokens
    # can differ under vLLM's atomic LoRA reductions. No study backend change.
    assert (fresh-repeated).abs().mean().item()<stale_gap/5
    assert stale_gap>0.2
    assert same.mean().item()<stale_gap/5
    for n,p in model.named_parameters():
        if n in saved:p.data.copy_(saved[n])
    eng.sync(2)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,betas=(0.9,0.99),eps=1e-8,weight_decay=0)
    for update in (1,2):
        previous=score(eng.lora)
        opt.zero_grad(set_to_none=True)
        loss=-hf_scores(grad=True).mean()
        loss.backward()
        params=[p for p in model.parameters() if p.requires_grad]
        finite([p.grad for p in params])
        torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        opt.param_groups[0]['lr']=update*1e-5
        before=[p.detach().clone() for p in params]
        opt.step();finite(params);finite(opt.state_dict())
        movement=sum((a-b).double().square().sum().item() for a,b in zip(params,before))
        assert movement>0
        eng.sync(update+2)
        adapter_hash=saved_identity(Path(eng.lora.lora_path))
        current=score(eng.lora)
        gap=hf_scores()-current
        propagated=(current-previous).abs().max().item()
        record=dict(update=update,adapter_sha256=adapter_hash,parameter_squared_movement=movement,
                    mean_abs_gap=gap.abs().mean().item(),signed_gap=gap.mean().item(),max_logprob_change=propagated)
        evidence['optimizer_updates'].append(record)
        print('OPTIMIZER PROPAGATION',json.dumps(record),flush=True)
        assert gap.abs().mean().item()<0.05
        assert abs(gap.mean().item())<0.01
        assert propagated>1e-7
    (tmp_path/'adapter-evidence.json').write_text(json.dumps(evidence,indent=2))
