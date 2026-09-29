"""
CPU unit tests for the GRPO pieces. The numbers are the worked examples from the design
discussion, so a failure here means the code no longer does what we said it does.
"""
import math

import pytest
import torch

from rl.grpo import group_advantages, grpo_loss, kl_k3, token_weights
from rl.logprobs import top_p_entropy
from rl.rollout import completion_masks, position_ids_from_mask


# ---------- advantages ----------

def adv_for(n_right, k=8, **kw):
    r = torch.tensor([1.0] * n_right + [0.0] * (k - n_right))
    return group_advantages(r, k, **kw)


@pytest.mark.parametrize("n_right,right,wrong", [
    (4, 1.0, -1.0), (3, 1.291, -0.775), (1, 2.646, -0.378), (7, 0.378, -2.646)])
def test_advantage_table(n_right, right, wrong):
    a = adv_for(n_right)
    assert a[0].item() == pytest.approx(right, abs=2e-3)       # eps=1e-4 shifts the 3rd decimal
    assert a[-1].item() == pytest.approx(wrong, abs=2e-3)
    assert a.sum().item() == pytest.approx(0, abs=1e-5)          # pushes up as much as down


@pytest.mark.parametrize("n_right", [0, 8])
def test_uniform_group_zero_advantage(n_right):
    assert torch.all(adv_for(n_right) == 0)                       # eps avoids 0/0


def test_advantage_without_std():
    a = adv_for(1, scale_std=False)
    assert a[0].item() == pytest.approx(0.875) and a[-1].item() == pytest.approx(-0.125)


def test_groups_are_independent():
    r = torch.tensor([1., 0, 0, 0, 1, 1, 1, 1])                   # two groups of 4
    a = group_advantages(r, 4)
    assert torch.all(a[4:] == 0) and a[0] > 0


# ---------- KL ----------

@pytest.mark.parametrize("p_now,p_ref,expected", [
    (0.3, 0.3, 0.0), (0.5, 0.25, 0.193), (0.25, 0.5, 0.307), (0.9, 0.1, 1.308)])
def test_kl_table(p_now, p_ref, expected):
    v = kl_k3(torch.tensor(math.log(p_now)), torch.tensor(math.log(p_ref))).item()
    assert v == pytest.approx(expected, abs=1e-3)


# ---------- loss + gradients ----------

def _loss_grad(logp_val, old_val, adv_val, **kw):
    logp = torch.tensor([[logp_val]], requires_grad=True)
    loss, info = grpo_loss(logp, torch.tensor([[old_val]]), None, torch.tensor([adv_val]),
                           torch.ones(1, 1), **kw)
    loss.backward()
    return loss.item(), logp.grad.item(), info


@pytest.mark.parametrize("adv,ratio,grad_zero", [
    (1.29, 1.5, True),    # good token already 50% likelier -> stop pushing
    (1.29, 0.5, False),   # good token got less likely -> keep pushing up
    (-0.77, 0.5, True),   # bad token already 50% less likely -> stop pushing
    (-0.77, 1.5, False),  # bad token got more likely -> keep pushing down
])
def test_clipping_table(adv, ratio, grad_zero):
    _, g, info = _loss_grad(math.log(ratio), 0.0, adv)
    assert (g == 0) == grad_zero
    assert bool(info["clipped"].item()) == grad_zero


def test_ratio_one_is_reinforce():
    """First update of a batch: old = logp.detach(), so ratio = 1 and the gradient must equal
    that of the plain REINFORCE loss  -sum w * A * logp."""
    torch.manual_seed(0)
    logp = torch.randn(4, 6).clamp(max=-0.01).requires_grad_()
    adv = torch.randn(4)
    mask = (torch.rand(4, 6) > 0.3).float()
    w = token_weights(mask, "seq")
    loss, _ = grpo_loss(logp, logp.detach(), None, adv, w)
    g1, = torch.autograd.grad(loss, logp)
    g2, = torch.autograd.grad(-(w * adv[:, None] * logp).sum(), logp)
    assert torch.allclose(g1, g2)


def test_uniform_group_zero_gradient():
    """8/8 right: every advantage is 0, and at step 0 the policy equals the reference,
    so the KL gradient is 0 too -> the whole gradient is exactly zero."""
    logp = torch.log(torch.rand(8, 5)).requires_grad_()
    adv = group_advantages(torch.ones(8), 8)
    loss, _ = grpo_loss(logp, logp.detach(), logp.detach(), adv, token_weights(torch.ones(8, 5), "token"),
                        beta=0.02)
    loss.backward()
    assert torch.all(logp.grad == 0)


def test_masked_tokens_get_no_gradient():
    logp = torch.log(torch.rand(2, 5)).requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    loss, _ = grpo_loss(logp, logp.detach(), None, torch.tensor([1.0, -1.0]), token_weights(mask, "token"))
    loss.backward()
    assert torch.all(logp.grad[mask == 0] == 0) and torch.all(logp.grad[mask == 1] != 0)


# ---------- aggregation (the 100- vs 400-token example) ----------

@pytest.mark.parametrize("agg,short_w,long_w", [
    ("seq", 1 / 200, 1 / 800), ("token", 1 / 500, 1 / 500), ("const", 1 / 1024, 1 / 1024)])
def test_token_weights(agg, short_w, long_w):
    mask = torch.zeros(2, 400)
    mask[0, :100] = 1
    mask[1, :] = 1
    w = token_weights(mask, agg, max_len=512)
    assert w[0, 0].item() == pytest.approx(short_w) and w[1, 0].item() == pytest.approx(long_w)
    assert w[0, 100:].sum() == 0


def test_microbatch_sum_equals_full_batch():
    torch.manual_seed(1)
    logp = torch.randn(6, 7).clamp(max=-0.01).requires_grad_()
    adv, mask = torch.randn(6), (torch.rand(6, 7) > 0.2).float()
    w = token_weights(mask, "seq")
    full, _ = grpo_loss(logp, logp.detach(), None, adv, w)
    parts = sum(grpo_loss(logp[i:i + 2], logp[i:i + 2].detach(), None, adv[i:i + 2], w[i:i + 2])[0]
                for i in range(0, 6, 2))
    assert full.item() == pytest.approx(parts.item(), rel=1e-6)


# ---------- entropy ----------

def test_entropy_full_vs_top_p():
    """0.5 and 0.4 on two tokens, 0.1 spread over the other 151,663: full ~2.14, top-p ~0.69."""
    V = 151_665
    p = torch.full((1, V), 0.1 / (V - 2), dtype=torch.float64)
    p[0, 0], p[0, 1] = 0.5, 0.4
    full = -(p * p.log()).sum().item()
    assert full == pytest.approx(2.14, abs=0.01)
    assert top_p_entropy(p, 0.85).item() == pytest.approx(0.687, abs=1e-3)   # keeps 2 tokens


def test_top_p_entropy_sharp_and_flat():
    assert top_p_entropy(torch.tensor([[1.0, 0.0, 0.0]]), 0.9).item() == pytest.approx(0)
    assert top_p_entropy(torch.full((1, 4), 0.25), 0.99).item() == pytest.approx(math.log(4))


# ---------- masks and positions ----------

EOS = [9, 8]


def test_completion_masks():
    comp = torch.tensor([
        [5, 6, 9, 8, 8],     # EOS at 2 -> length 3 (EOS included), rest is padding
        [5, 8, 8, 8, 8],     # the pad id is also an EOS id: the first one ends the answer
        [5, 6, 7, 5, 6],     # no EOS -> truncated, length 5
        [9, 8, 8, 8, 8],     # immediate EOS -> length 1
    ])
    mask, lengths, trunc = completion_masks(comp, EOS)
    assert lengths.tolist() == [3, 2, 5, 1]
    assert trunc.tolist() == [False, False, True, False]
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]


def test_position_ids_left_padding():
    am = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    assert position_ids_from_mask(am).tolist() == [[0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5]]
