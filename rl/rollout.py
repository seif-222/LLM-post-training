"""
Rollout: sample K answers per question and package them for the loss.

Layout of one batch row (P prompt columns, T completion columns):

    [pad pad <prompt tokens>][<answer tokens> <eos> pad pad]
     <-- left-padded, P --->  <------- T columns -------->

  attention_mask   0 on padding, 1 on every real token: prompt, answer and the first EOS
  completion_mask  1 only on answer tokens through the first EOS (the tokens we train on)

The first EOS is included: ending the answer is a decision the model makes, so it gets the
same credit or blame as every other token. Everything after it is padding. For Qwen, the pad
id is also one of the EOS ids, so "first EOS" is the only safe cut.
"""
import torch

from rl.data import build_prompt


def completion_masks(completion_ids, eos_ids):
    """completion_ids [N, T] as returned by generate (after the prompt columns).
    -> mask [N, T] (1 through the first EOS inclusive), lengths [N], truncated [N] (no EOS)."""
    N, T = completion_ids.shape
    is_eos = torch.isin(completion_ids, torch.tensor(eos_ids, device=completion_ids.device))
    idx = torch.arange(T, device=completion_ids.device).expand(N, T)
    first = torch.where(is_eos, idx, torch.full_like(idx, T)).min(dim=1).values   # T if no EOS
    truncated = first == T
    lengths = torch.where(truncated, first, first + 1)
    mask = (idx < lengths.unsqueeze(1)).long()
    return mask, lengths, truncated


def position_ids_from_mask(attention_mask):
    """Positions count real tokens only: [0 0 1 2 3] for mask [0 1 1 1 1], the same positions
    generate() used. torch.arange would give [0 1 2 3 4], shifting every real token of a
    left-padded row by the same constant. That is harmless for RoPE, which only sees distances
    (tests/test_gpu_logprobs.py checks it), but wrong for absolute position embeddings
    (GPT-2), so we match generate() exactly."""
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


@torch.no_grad()
def rollout(model, tok, problems, k, gen_cfg, prompt_style="boxed", max_seqs=256, sample_logprobs=False):
    """Sample k answers for each problem.

    Returns a dict of CPU tensors plus python lists; row i*k + j is sample j of problem i:
      input_ids, attention_mask, completion_mask  [N, P+T]
      lengths, truncated                          [N]
      texts                                       list[str], N
      prompt_len                                  P (int)
      sample_logp  [N, T] log p(token) as seen *during sampling* (only if sample_logprobs;
                   keeps [N, T, V] logits around, so tests only)
    """
    eos_ids = gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id]
    gen_cfg.num_return_sequences = k
    gen_cfg.return_dict_in_generate = True
    gen_cfg.output_logits = sample_logprobs
    prompts = [build_prompt(tok, p["question"], prompt_style) for p in problems]
    # Pad every chunk to the same prompt width so the chunks can be concatenated.
    enc_all = tok(prompts, return_tensors="pt", padding=True)
    P = enc_all["input_ids"].shape[1]

    per_call = max(1, max_seqs // k)
    ids, amask, cmask, lengths, truncated, slp = [], [], [], [], [], []
    for i in range(0, len(prompts), per_call):
        enc = {key: v[i:i + per_call].to(model.device) for key, v in enc_all.items()}
        out = model.generate(**enc, generation_config=gen_cfg)
        seqs = out.sequences                                               # [n*k, P+T_i]
        comp = seqs[:, P:]
        if sample_logprobs:
            logits = torch.stack(out.logits, 1).float() / (gen_cfg.temperature or 1.0)
            slp.append(logits.log_softmax(-1).gather(-1, comp.unsqueeze(-1)).squeeze(-1).cpu())
        cm, ln, tr = completion_masks(comp, eos_ids)
        pm = enc["attention_mask"].repeat_interleave(k, dim=0)             # rows are grouped by prompt
        ids.append(seqs.cpu()); amask.append(torch.cat([pm, cm], 1).cpu())
        cmask.append(torch.cat([torch.zeros_like(pm), cm], 1).cpu())
        lengths.append(ln.cpu()); truncated.append(tr.cpu())

    # Chunks can end at different T; right-pad them to the widest.
    width = max(x.shape[1] for x in ids)
    pad = lambda x, v: torch.nn.functional.pad(x, (0, width - x.shape[1]), value=v)
    input_ids = torch.cat([pad(x, tok.pad_token_id) for x in ids])
    attention_mask = torch.cat([pad(x, 0) for x in amask])
    completion_mask = torch.cat([pad(x, 0) for x in cmask])
    texts = tok.batch_decode(input_ids[:, P:], skip_special_tokens=True)
    out = {
        "input_ids": input_ids, "attention_mask": attention_mask, "completion_mask": completion_mask,
        "lengths": torch.cat(lengths), "truncated": torch.cat(truncated), "texts": texts, "prompt_len": P,
    }
    if sample_logprobs:
        pad_t = lambda x: torch.nn.functional.pad(x, (0, width - P - x.shape[1]))
        out["sample_logp"] = torch.cat([pad_t(x) for x in slp]) * completion_mask[:, P:]
    return out
