"""
Answer generators ("engines"). The trainer only talks to these two methods:

  sample(problems, k, temperature)  -> the rollout dict (same layout as rl.rollout.rollout)
  greedy(problems)                  -> list of {text, length, truncated}, one per problem

and calls sync(step) after every optimizer step so the engine samples from the current policy.

  HFEngine    Hugging Face generate() on the training model itself (LoRA active). Slow
              (~2,200 tok/s on the laptop, half of it KV-cache copying) but exact: it is the
              model we train. The reference for every check.
  VLLMEngine  vLLM with its own copy of the frozen base model. After each step we save the
              LoRA adapter (B and A, 35 MB) and hand vLLM a new LoRARequest, so it samples
              from the current policy. Much faster, but its numbers differ slightly from the
              trainer's (training-inference mismatch), so it also returns the log-prob of every
              sampled token, and the trainer logs the gap.

vLLM trap: by default it copies Qwen's generation_config (top_k=20, top_p=0.8, repetition
penalty). generation_config="vllm" plus explicit neutral SamplingParams turn that off.
"""
import os
import tempfile

import torch

from rl.data import build_prompt, eos_ids_of
from rl.campaign import isolated_rng
from rl.eval import gen_config, generate
from rl.rollout import completion_masks, rollout


def pack_completions(prompt_ids, comps, lps, finished, k, tok, eos_ids):
    """Plain token lists -> the rl.rollout.rollout dict (pure function, so it's testable without vLLM).
    prompt_ids: one id list per problem; comps / lps / finished: one per sample, grouped k per
    problem (row i*k + j = sample j of problem i). Prompts are left-padded to a common width P,
    answers right-padded, masks cover the answer through the first EOS."""
    P = max(len(p) for p in prompt_ids)
    T = max(1, max(len(c) for c in comps))
    N, pad = len(comps), tok.pad_token_id
    prompt = torch.full((N, P), pad, dtype=torch.long)
    pmask = torch.zeros((N, P), dtype=torch.long)
    comp = torch.full((N, T), pad, dtype=torch.long)
    slp = torch.zeros((N, T))
    for i, (c, lp) in enumerate(zip(comps, lps)):
        p = prompt_ids[i // k]
        prompt[i, P - len(p):], pmask[i, P - len(p):] = torch.tensor(p), 1
        if c:
            comp[i, :len(c)], slp[i, :len(c)] = torch.tensor(c), torch.tensor(lp, dtype=torch.float)
    cm, lengths, truncated = completion_masks(comp, eos_ids)
    # A row that stopped early must end in an EOS we can see, or the masks would be wrong.
    assert torch.equal(torch.tensor(finished), ~truncated), "stop reasons disagree with the EOS scan"
    return {
        "input_ids": torch.cat([prompt, comp], 1),
        "attention_mask": torch.cat([pmask, cm], 1),
        "completion_mask": torch.cat([torch.zeros_like(pmask), cm], 1),
        "lengths": lengths, "truncated": truncated,
        "texts": tok.batch_decode(comp, skip_special_tokens=True),
        "prompt_len": P,
        "sample_logp": slp * cm,
    }


class HFEngine:
    def __init__(self, model, tok, max_new_tokens, prompt_style="boxed", gen_batch=256):
        self.model, self.tok = model, tok
        self.max_new_tokens, self.prompt_style, self.gen_batch = max_new_tokens, prompt_style, gen_batch
        self.eos_ids = eos_ids_of(model.generation_config)

    def sync(self, step):
        pass                                        # samples straight from the live training model

    def release(self):
        pass

    def sample(self, problems, k, temperature, requests=None):
        if requests is not None:
            if len(requests) != len(problems):
                raise ValueError("request count mismatch")
            prompts, comps, finished, logprobs = [], [], [], []
            has_logprobs = False
            for problem, request in zip(problems, requests):
                with isolated_rng(request["seed"]):
                    ro = self.sample([problem], k, temperature)
                P = ro["prompt_len"]
                prompts.append(ro["input_ids"][0, :P][ro["attention_mask"][0, :P].bool()].tolist())
                for i in range(k):
                    comps.append(ro["input_ids"][i, P:P+int(ro["lengths"][i])].tolist())
                    finished.append(not bool(ro["truncated"][i]))
                    has_logprobs = "sample_logp" in ro
                    logprobs.append(ro["sample_logp"][i, :int(ro["lengths"][i])].tolist() if has_logprobs else [0.] * int(ro["lengths"][i]))
            packed = pack_completions(prompts, comps, logprobs, finished, k, self.tok, self.eos_ids)
            if not has_logprobs:
                del packed["sample_logp"]
            packed["requests"] = requests
            return packed
        cfg = gen_config(self.model, True, temperature, self.max_new_tokens)
        return rollout(self.model, self.tok, problems, k, cfg, self.prompt_style, self.gen_batch)

    def greedy(self, problems, batch_size=200):
        prompts = [build_prompt(self.tok, p["question"], self.prompt_style) for p in problems]
        gens = generate(self.model, self.tok, prompts, gen_config(self.model, False, None, self.max_new_tokens),
                        batch_size=batch_size, verbose=False)
        return [g[0] for g in gens]


class VLLMEngine:
    """model: the peft training model (its adapter is saved and handed over at every sync), or
    None for evaluation, where `adapter_dir` is an already-saved adapter (or None = base model)."""

    def __init__(self, model, tok, model_name, max_new_tokens, prompt_style="boxed", adapter_dir=None,
                 gpu_memory_utilization=0.3, max_lora_rank=64, seed=0, sleep=False, max_loras=1, revision=None):
        # Run vLLM's engine inside this process. The trainer has already started CUDA, and a
        # separate engine process would then have to be spawned (re-importing the caller).
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # vLLM's default FlashInfer sampler compiles a CUDA kernel on first use, and the image has
        # no CUDA compiler (nvcc). The PyTorch sampler needs none; with top-p 1 / no top-k it just
        # samples from the full softmax.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from transformers import GenerationConfig
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        self.model, self.tok = model, tok
        self.max_new_tokens, self.prompt_style = max_new_tokens, prompt_style
        self.eos_ids = eos_ids_of(model.generation_config if model is not None
                                  else GenerationConfig.from_pretrained(model_name, revision=revision))
        self.adapter_dir, self.sleep_between = adapter_dir, sleep
        self.llm = LLM(model=model_name, revision=revision, tokenizer_revision=revision, dtype="bfloat16", enable_lora=True, max_lora_rank=max_lora_rank,
                       max_loras=max_loras, gpu_memory_utilization=gpu_memory_utilization, seed=seed,
                       generation_config="vllm", max_model_len=1024, enable_sleep_mode=sleep)
        self.lora, self.asleep = None, False
        if model is not None:
            self.sync(0)
        elif adapter_dir:
            self.lora = LoRARequest("eval", 1, adapter_dir)

    def sync(self, step):
        """Hand vLLM the current adapter. A new id AND a new folder per step, so no cache inside
        vLLM (by id, name or path) can hand back an older adapter. The previous folder is deleted."""
        import shutil
        from vllm.lora.request import LoRARequest
        path = os.path.join(self.adapter_dir, f"step{step}")
        self.model.save_pretrained(path)
        if self.lora is not None and self.lora.lora_path != path:
            shutil.rmtree(self.lora.lora_path, ignore_errors=True)
        self.lora = LoRARequest(f"step{step}", step + 1, path)

    def _wake(self):
        if self.asleep:
            self.llm.wake_up()
            self.asleep = False

    def release(self):
        """Laptop mode: free vLLM's GPU memory while the trainer runs."""
        if self.sleep_between and not self.asleep:
            self.llm.sleep(level=1)
            self.asleep = True

    def _params(self, n, temperature, logprobs, seed=None):
        from vllm import SamplingParams
        return SamplingParams(n=n, temperature=temperature, seed=seed, top_p=1.0, top_k=-1, min_p=0.0,
                              repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                              max_tokens=self.max_new_tokens, stop_token_ids=self.eos_ids,
                              logprobs=0 if logprobs else None, skip_special_tokens=True)

    def _run(self, problems, params):
        from vllm.inputs import TokensPrompt
        self._wake()
        prompt_ids = [self.tok(build_prompt(self.tok, p["question"], self.prompt_style))["input_ids"] for p in problems]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], params,
                                 lora_request=self.lora, use_tqdm=False)
        return prompt_ids, outs

    def sample(self, problems, k, temperature, requests=None):
        """Same dict as rl.rollout.rollout, plus sample_logp [N, T] (vLLM's log p of each sampled token)."""
        if requests is not None and len(requests) != len(problems):
            raise ValueError("request count mismatch")
        params = ([self._params(k, temperature, True, req["seed"]) for req in requests]
                  if requests is not None else self._params(k, temperature, True))
        prompt_ids, outs = self._run(problems, params)
        comps, lps, finished = [], [], []
        for out in outs:                                      # rows grouped by prompt: row i*k + j
            assert len(out.outputs) == k
            for c in out.outputs:
                ids = list(c.token_ids)
                comps.append(ids)
                lps.append([d[t].logprob for t, d in zip(ids, c.logprobs)])
                finished.append(c.finish_reason == "stop")
        packed = pack_completions(prompt_ids, comps, lps, finished, k, self.tok, self.eos_ids)
        if requests is not None:
            packed["requests"] = requests
        return packed

    def greedy(self, problems, batch_size=None):
        _, outs = self._run(problems, self._params(1, 0.0, logprobs=False))
        rows = []
        for out in outs:
            c = out.outputs[0]
            ids = torch.tensor([list(c.token_ids) or [self.tok.pad_token_id]])
            _, lengths, truncated = completion_masks(ids, self.eos_ids)
            assert (c.finish_reason == "stop") == (not bool(truncated[0])), "stop reason disagrees with the EOS scan"
            rows.append({"text": self.tok.decode(c.token_ids, skip_special_tokens=True),
                         "length": int(lengths[0]) if c.token_ids else 0, "truncated": bool(truncated[0]),
                         "token_ids": list(c.token_ids), "finish_reason": c.finish_reason})
        return rows


def make_engine(kind, model, tok, args, out_dir):
    if kind == "hf":
        return HFEngine(model, tok, args.max_new_tokens, args.prompt, args.gen_batch)
    if kind == "vllm":
        # The per-step hand-over copy of the adapter goes to local scratch, not the run folder
        # (on Modal that is the shared disk, and it would be re-uploaded at every checkpoint).
        return VLLMEngine(model, tok, args.model, args.max_new_tokens, args.prompt,
                          adapter_dir=tempfile.mkdtemp(prefix="vllm_adapter_"),
                          gpu_memory_utilization=args.vllm_mem, max_lora_rank=max(8, args.lora_r), seed=args.seed,
                          sleep=args.vllm_sleep, revision=getattr(args, "model_revision", None))
    raise ValueError(kind)
