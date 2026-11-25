"""
The GRPO objective, one piece per function, so each piece can be tested on its own.

Per answer i in a group of K answers to the same question:
    A_i = (r_i - mean(r)) / (std(r) + eps)            advantage; std division is switchable

Per answer token t:
    ratio  = p_now(t) / p_sampled(t)                   = 1 on the first update of a batch
    pg     = -min(ratio * A, clip(ratio, 1-eps_low, 1+eps_high) * A)
    kl     = e^d - d - 1,  d = log p_ref(t) - log p_now(t)     (k3 estimator, always >= 0)
    loss_t = pg + beta * kl - ent_coef * entropy_t

Then loss = sum over tokens of w_t * loss_t, where the weights w_t decide how tokens are
averaged (see token_weights). The weights are computed for the whole batch up front, so
summing the loss over micro-batches gives exactly the full-batch gradient.
"""
import torch


def group_advantages(rewards, k, scale_std=True, eps=1e-4):
    """rewards [N], rows grouped k at a time (row i*k + j = sample j of question i) -> [N].
    Uses the population std (divide by k), matching the worked examples: 4/8 right -> +1 / -1."""
    r = rewards.view(-1, k).float()
    a = r - r.mean(1, keepdim=True)
    if scale_std:
        a = a / (r.std(1, keepdim=True, correction=0) + eps)
    return a.view(-1)


def token_weights(mask, agg, max_len=None, n_seqs=None, n_tokens=None):
    """mask [N, T] -> weights [N, T]; sum(weights) sets the loss scale.
    The averages must be over the WHOLE batch. For a micro-batch, pass the whole batch's
    n_seqs / n_tokens; they default to this mask's own totals.
      seq    average within each answer, then across answers (original GRPO).
             A token of a 400-token answer counts 4x less than one of a 100-token answer.
      token  every token in the batch counts the same (DAPO).
      const  divide by a fixed N * max_len (Dr. GRPO): every token counts the same, and the
             scale doesn't change with how long this batch's answers happen to be."""
    mask = mask.float()
    n = n_seqs or mask.shape[0]
    if agg == "seq":
        return mask / (mask.sum(1, keepdim=True).clamp(min=1) * n)
    if agg == "token":
        return mask / (n_tokens or mask.sum().clamp(min=1))
    if agg == "const":
        assert max_len, "const aggregation needs max_len"
        return mask / (n * max_len)
    raise ValueError(agg)


def kl_k3(logp, ref_logp):
    d = ref_logp - logp
    return torch.exp(d) - d - 1


def tis_weights(old_logp, sample_logp, cap):
    """Truncated importance sampling. The answers were sampled by vLLM, whose probabilities
    differ slightly from the trainer's. Reweight each token by p_trainer / p_vllm, capped at
    `cap` so one odd token can't blow up the update. Detached: it scales the gradient, it isn't
    trained. Example, cap 2: p_trainer 0.30 vs p_vllm 0.25 -> 1.2; 0.30 vs 0.10 -> 3.0 -> 2."""
    return torch.exp(old_logp - sample_logp).clamp(max=cap).detach()


def grpo_loss(logp, old_logp, ref_logp, adv, weights, clip_low=0.2, clip_high=0.2,
              beta=0.0, entropy=None, ent_coef=0.0, tis=None):
    """logp/old_logp/ref_logp/weights [B, T], adv [B]. ref_logp may be None when beta == 0.
    tis [B, T]: optional per-token correction weights (tis_weights) on the policy part.
    Returns (scalar loss, dict for logging: detached per-token tensors + the loss split into
    its policy part and its KL part)."""
    A = adv.unsqueeze(1)
    ratio = torch.exp(logp - old_logp)
    clipped_ratio = ratio.clamp(1 - clip_low, 1 + clip_high)
    pg = -torch.min(ratio * A, clipped_ratio * A)
    if tis is not None:
        pg = pg * tis
    per_token = pg
    kl = kl_k3(logp, ref_logp) if ref_logp is not None else torch.zeros_like(logp)
    if beta:
        per_token = per_token + beta * kl
    if ent_coef:
        per_token = per_token - ent_coef * entropy
    loss = (weights * per_token).sum()
    # The clipped branch is active when the token already moved far enough in A's direction.
    is_clipped = ((ratio > 1 + clip_high) & (A > 0)) | ((ratio < 1 - clip_low) & (A < 0))
    return loss, {"kl": kl.detach(), "clipped": is_clipped.float(), "ratio": ratio.detach(),
                  "pg_part": (weights * pg).sum().item(), "kl_part": beta * (weights * kl).sum().item()}


def mask_advantages(advantages, mode="full"):
    if mode == "full":
        return advantages
    if mode == "positive":
        return advantages.clamp(min=0)
    if mode == "negative":
        return advantages.clamp(max=0)
    raise ValueError(mode)


def gradient_diagnostics(total, weighted_kl):
    """Called on detached accumulated vectors, before clipping; zero ratios are missing."""
    policy = [t - k for t, k in zip(total, weighted_kl)]
    def norm(xs):
        return sum(x.double().square().sum() for x in xs).sqrt().item()
    p, k, t = norm(policy), norm(weighted_kl), norm(total)
    dot = sum((a.double() * b.double()).sum() for a, b in zip(policy, weighted_kl)).item()
    return {"policy_grad_norm": p, "weighted_kl_grad_norm": k, "total_grad_norm": t,
            "policy_kl_dot": dot, "policy_kl_alignment": dot / (p * k) if p and k else None,
            "kl_policy_grad_ratio": k / p if p else None}
