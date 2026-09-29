"""
Training-time token log-probs (and entropies) for the completion tokens.

Sampling gives token ids but no gradient. To train, we run prompt + answer back through the
model in one forward pass and read off log p(token) at every answer position.

Two memory tricks, because the vocab has 151,936 entries:
  1. Run only the transformer body, then apply the output layer (lm_head) to the answer
     positions only. Prompt and padding positions never become logits.
  2. Do that in chunks of positions, each wrapped in activation checkpointing, so the
     [chunk, 151936] float32 logits are rebuilt during backward, not stored.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rl.rollout import position_ids_from_mask


def _base(model):
    """The HF causal-LM under an optional peft wrapper. LoRA lives inside its linear layers,
    so calling the base model directly still uses (or, under disable_adapter, skips) LoRA."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def top_p_entropy(probs, top_p):
    """Entropy of the smallest set of tokens covering probability top_p, renormalized.
    probs [n, V] -> [n]. A token is kept if the mass *before* it is still below top_p, so the
    token that crosses the threshold is included (the same rule top-p sampling uses)."""
    p_sorted, _ = probs.sort(dim=-1, descending=True)
    before = p_sorted.cumsum(-1) - p_sorted
    q = p_sorted * (before < top_p)
    q = q / q.sum(-1, keepdim=True)
    return -(q * torch.log(q.clamp_min(1e-30))).sum(-1)


def _chunk_stats(h, targets, weight, temperature, top_p, entropy_grad):
    """h [n, H] hidden states, targets [n] -> logp [n], full entropy [n], top-p entropy [n]."""
    logits = (h @ weight.T).float() / temperature
    logz = logits.logsumexp(-1)
    logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logz
    with torch.set_grad_enabled(entropy_grad and torch.is_grad_enabled()):
        probs = torch.exp(logits - logz.unsqueeze(-1))
        ent = logz - (probs * logits).sum(-1)                      # -sum p log p, rewritten stably
        ent_top = top_p_entropy(probs, top_p) if top_p is not None else torch.zeros_like(ent)
    return logp, ent, ent_top


def token_logprobs(model, input_ids, attention_mask, completion_mask, prompt_len,
                   temperature=1.0, top_p=0.9, entropy_grad=False, chunk=256):
    """-> dict of [B, T] tensors aligned with input_ids[:, prompt_len:] (zeros off the mask):
         logp       log p(token) under the current model (with grad if grad is enabled)
         entropy    full-vocab entropy of the next-token distribution at that position
         entropy_top  top-p entropy (see top_p_entropy)
    Entropies carry grad only if entropy_grad (needed only when the entropy bonus is on)."""
    B, L = input_ids.shape
    P, T = prompt_len, L - prompt_len
    device = input_ids.device

    # Trim columns that are padding in every row. Left trim leaves positions unchanged,
    # because positions come from the attention mask, not the column index.
    real = attention_mask.any(0).nonzero()
    lo = min(int(real[0]), P - 1)
    hi = max(int(completion_mask.any(0).nonzero()[-1]) + 1, P + 1)
    ids, am = input_ids[:, lo:hi], attention_mask[:, lo:hi]
    cm = completion_mask[:, P:hi]                                  # [B, hi-P]

    base = _base(model)
    h = base.model(input_ids=ids, attention_mask=am, position_ids=position_ids_from_mask(am),
                   use_cache=False).last_hidden_state
    # The hidden state at column c predicts the token at column c+1.
    h = h[:, P - 1 - lo: hi - 1 - lo]                              # predicts columns P .. hi-1
    targets = ids[:, P - lo:]
    sel = cm.bool()
    h_sel, t_sel = h[sel], targets[sel]                            # only real answer tokens

    weight = base.lm_head.weight
    outs = [[], [], []]
    for i in range(0, h_sel.shape[0], chunk):
        args = (h_sel[i:i + chunk], t_sel[i:i + chunk], weight, temperature, top_p, entropy_grad)
        res = checkpoint(_chunk_stats, *args, use_reentrant=False) if torch.is_grad_enabled() \
            else _chunk_stats(*args)
        for o, r in zip(outs, res):
            o.append(r)

    sel_full = F.pad(sel, (0, T - (hi - P)))                        # back to all T answer columns
    zeros = torch.zeros(B, T, device=device, dtype=torch.float32)
    return {name: zeros.masked_scatter(sel_full, torch.cat(o))
            for name, o in zip(("logp", "entropy", "entropy_top"), outs)}
