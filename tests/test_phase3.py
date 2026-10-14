"""CPU tests for the Phase 3 additions: pass@k, the vLLM correction weights, the loss split,
packing vLLM outputs into the rollout layout, checkpoints and resume bookkeeping."""
import gzip
import json
import math
import os

import pytest
import torch

from rl.engine import pack_completions
from rl.eval import pass_at_k, passk_summary
from rl.grpo import grpo_loss, tis_weights, token_weights
from rl.rollout import completion_masks
import train_grpo as tg

EOS, PAD = 9, 9          # like Qwen: the pad id is also an EOS id


class FakeTok:
    pad_token_id = PAD

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(t) for t in row.tolist() if t != PAD) for row in ids]


# ---------- pass@k ----------

@pytest.mark.parametrize("n,c,k,want", [
    (8, 0, 1, 0.0), (8, 0, 8, 0.0),          # never right -> 0 for every k
    (8, 8, 1, 1.0),                          # always right -> 1
    (8, 1, 1, 1 / 8), (8, 1, 8, 1.0),        # one right answer out of 8
    (8, 4, 2, 1 - 6 / 28),                   # C(4,2)/C(8,2) = 6/28 chance both tries are wrong
])
def test_pass_at_k_table(n, c, k, want):
    assert pass_at_k(n, c, k) == pytest.approx(want)


def test_pass_at_k_matches_combinations():
    for n in (4, 8, 16):
        for c in range(n + 1):
            for k in range(1, n + 1):
                exact = 1 - math.comb(n - c, k) / math.comb(n, k)
                assert pass_at_k(n, c, k) == pytest.approx(exact)


def test_passk_summary():
    s = passk_summary([0, 8, 4], 8)
    assert list(s) == ["pass@1", "pass@2", "pass@4", "pass@8"]
    assert s["pass@1"] == pytest.approx((0 + 1 + 0.5) / 3)
    assert s["pass@8"] == pytest.approx(2 / 3)


# ---------- vLLM correction weights and the loss split ----------

def test_tis_weights_example():
    old = torch.log(torch.tensor([[0.30, 0.30, 0.30]]))
    samp = torch.log(torch.tensor([[0.25, 0.10, 0.60]]))
    assert tis_weights(old, samp, 2.0).tolist()[0] == pytest.approx([1.2, 2.0, 0.5])


def test_tis_of_ones_changes_nothing():
    logp = torch.randn(3, 5, requires_grad=True)
    adv, w = torch.tensor([1.0, -0.5, 0.2]), token_weights(torch.ones(3, 5), "token")
    a, _ = grpo_loss(logp, logp.detach(), None, adv, w)
    b, _ = grpo_loss(logp, logp.detach(), None, adv, w, tis=torch.ones(3, 5))
    assert a.item() == pytest.approx(b.item())


def test_loss_is_policy_part_plus_kl_part():
    torch.manual_seed(0)
    logp = torch.randn(4, 6) - 2
    ref = logp + 0.1 * torch.randn(4, 6)
    w = token_weights(torch.ones(4, 6), "seq")
    loss, info = grpo_loss(logp, logp, ref, torch.tensor([1.0, -1.0, 0.5, -0.5]), w, beta=0.05)
    assert loss.item() == pytest.approx(info["pg_part"] + info["kl_part"], abs=1e-6)
    assert info["kl_part"] > 0


# ---------- packing vLLM outputs into the rollout layout ----------

def test_pack_completions_layout():
    prompts = [[1, 2, 3], [4, 5]]                           # 2 problems, k = 2
    comps = [[6, EOS], [7, 7, 7], [8, EOS], [6, 6, EOS]]
    lps = [[-0.1, -0.2], [-1.0, -1.0, -1.0], [-0.3, -0.4], [-0.5, -0.5, -0.6]]
    finished = [True, False, True, True]
    ro = pack_completions(prompts, comps, lps, finished, 2, FakeTok(), [EOS])
    P = ro["prompt_len"]
    assert P == 3 and ro["input_ids"].shape == (4, 6)
    assert ro["input_ids"][2].tolist() == [PAD, 4, 5, 8, EOS, PAD]          # left-padded prompt
    assert ro["attention_mask"][2].tolist() == [0, 1, 1, 1, 1, 0]
    assert ro["completion_mask"][2].tolist() == [0, 0, 0, 1, 1, 0]          # through the first EOS
    assert ro["lengths"].tolist() == [2, 3, 2, 3]
    assert ro["truncated"].tolist() == [False, True, False, False]
    assert ro["sample_logp"][0].tolist() == pytest.approx([-0.1, -0.2, 0.0])
    # The same masks rollout() would build from the raw answer columns.
    cm, _, _ = completion_masks(ro["input_ids"][:, P:], [EOS])
    assert torch.equal(cm, ro["completion_mask"][:, P:])


def test_pack_completions_rejects_hidden_eos():
    """A 'stop' with no EOS token in the ids would silently break the masks -> must fail loudly."""
    with pytest.raises(AssertionError):
        pack_completions([[1]], [[6, 7]], [[-0.1, -0.1]], [True], 1, FakeTok(), [EOS])


def test_split_and_collate_carry_sample_logp():
    ro = pack_completions([[1, 2]], [[6, EOS], [7, 8, EOS]], [[-0.1, -0.2], [-0.3, -0.4, -0.5]],
                          [True, True], 2, FakeTok(), [EOS])
    group = tg.split_rows(ro, 2)[0]
    assert group[1]["sample_logp"].tolist() == pytest.approx([-0.3, -0.4, -0.5])
    batch, P, slp = tg.collate(group, PAD, "cpu")
    assert slp.shape == batch["input_ids"][:, P:].shape
    assert slp[0].tolist() == pytest.approx([-0.1, -0.2, 0.0])


# ---------- checkpoints and resume bookkeeping ----------

def test_question_stream_resume():
    a = tg.QuestionStream(list(range(10)), seed=3)
    a.next(7)
    saved = a.state_dict()
    want = a.next(12)                                        # crosses an epoch boundary
    b = tg.QuestionStream(list(range(10)), seed=99)
    b.load_state_dict(saved)
    assert b.next(12) == want and b.seen == a.seen


def test_trim_logs(tmp_path):
    rows = [{"type": "train", "step": s} for s in (1, 2, 3, 4)]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with gzip.open(tmp_path / "samples.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with gzip.open(tmp_path / "samples.jsonl.gz", "at") as f:  # a second gzip member, as appends make
        f.write(json.dumps({"step": 5}) + "\n")
    tg.trim_logs(str(tmp_path), 2)
    assert [json.loads(l)["step"] for l in open(tmp_path / "metrics.jsonl")] == [1, 2]
    with gzip.open(tmp_path / "samples.jsonl.gz", "rt") as f:
        assert [json.loads(l)["step"] for l in f] == [1, 2]


def tiny_lora_model():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen2Config
    cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    return get_peft_model(AutoModelForCausalLM.from_config(cfg),
                          LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", lora_dropout=0.0))


def test_checkpoint_round_trip(tmp_path):
    model = tiny_lora_model()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
    ids = torch.randint(0, 32, (2, 8))
    for _ in range(3):                                       # so B is non-zero and Adam has state
        opt.zero_grad()
        model(input_ids=ids, labels=ids).loss.backward()
        opt.step()
        sched.step()
    stream, gen = tg.QuestionStream(list(range(5)), 0), torch.Generator().manual_seed(1)
    stream.next(3)
    path = str(tmp_path / "ckpt")
    tg.save_checkpoint(path, model, opt, sched, stream, gen, 3, {"elapsed": 12.5, "wandb_id": "abc"})
    tg.save_checkpoint(path, model, opt, sched, stream, gen, 3, {"elapsed": 12.5, "wandb_id": "abc"})  # overwrite
    want = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    want_lr = sched.get_last_lr()[0]

    fresh = tiny_lora_model()                                # B = 0 again
    params2 = [p for p in fresh.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(params2, lr=1e-2)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: min(1.0, (s + 1) / 10))
    stream2, gen2 = tg.QuestionStream(list(range(5)), 7), torch.Generator()
    st = tg.load_checkpoint(path, fresh, opt2, sched2, stream2, gen2)
    got = {n: p for n, p in fresh.named_parameters() if p.requires_grad}
    assert all(torch.equal(want[n], got[n]) for n in want)
    assert st["step"] == 3 and st["elapsed"] == 12.5 and st["wandb_id"] == "abc"
    assert sched2.get_last_lr()[0] == pytest.approx(want_lr)
    assert stream2.next(4) == stream.next(4)
    assert torch.equal(gen2.get_state(), gen.get_state())
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert all(torch.equal(s1[i]["exp_avg"], s2[i]["exp_avg"]) for i in s1)
    assert not os.path.exists(path + ".tmp") and not os.path.exists(path + ".old")


def test_warmup_schedule():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
    lrs = []
    for _ in range(12):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-5)          # step 1: lr / 10
    assert lrs[9] == pytest.approx(1e-4)          # step 10: full lr
    assert lrs[11] == pytest.approx(1e-4)         # then constant
