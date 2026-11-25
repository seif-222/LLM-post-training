"""
GSM8K loading and prompt building.

Each problem is {"question": str, "answer": float, "solution": str}.
`solution` is the reference worked answer (ends in "#### <n>"); RL never trains on it,
it's kept only for inspecting failures.
"""
import re
import math
from datasets import load_dataset

# Two answer formats. "hash" matches GSM8K's own "#### <n>"; "boxed" is what Qwen2.5 was
# tuned on for math ("\boxed{}"). Which one the prompt asks for changes baseline accuracy,
# and the reward parser accepts both.
PROMPTS = {
    "hash": ("Solve the following math problem step by step. "
             "At the end, write the final answer as a number on its own line in the form "
             "'#### <number>'.\n\n{question}"),
    "boxed": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
}


# Qwen2.5 chat format ends a turn with <|im_end|>. The Instruct model lists it as an EOS id; the raw
# base model only lists <|endoftext|>, so it would run past the end of its answer (checked: 22% of
# its answers hit the 512 limit, padded with junk). Every model we use stops at both.
CHAT_END_ID = 151645            # <|im_end|>


def eos_ids_of(generation_config):
    e = generation_config.eos_token_id
    e = e if isinstance(e, list) else [e]
    return e + [CHAT_END_ID] * (CHAT_END_ID not in e)


def extract_final_answer(solution_text):
    """GSM8K solutions end with '#### <number>' -- pull that number out as a float."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", solution_text)
    assert match, f"couldn't find '#### <number>' in: {solution_text!r}"
    value = float(match.group(1).replace(',', ''))
    if not math.isfinite(value):
        raise ValueError('nonfinite gold answer')
    return value


def load_gsm8k(split, revision=None):
    """split: 'train' (7,473) or 'test' (1,319)."""
    ds = load_dataset("openai/gsm8k", "main", split=split, revision=revision)
    if revision and (not ds.cache_files or any(revision not in f["filename"] for f in ds.cache_files)):
        raise ValueError("GSM8K cache does not prove the requested pinned revision")
    return [{"id": i, "question": r["question"], "answer": extract_final_answer(r["answer"]), "solution": r["answer"]}
            for i, r in enumerate(ds)]


def build_prompt(tokenizer, question, style="hash"):
    """Chat-formatted prompt string ending at the start of the assistant turn.
    style "plain": the boxed instruction with NO chat format, for the raw base model. With the chat
    format the base model never emits <|im_end|>; it writes fake turns and a new question after its
    answer (40 test questions: 10% strict, 22% truncated). Plain: 40% strict, 85% boxed, 10% truncated.
    Dr. GRPO found the same: Qwen2.5 base models do best with no template."""
    if style == "plain":
        return PROMPTS["boxed"].format(question=question) + "\n"
    messages = [{"role": "user", "content": PROMPTS[style].format(question=question)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
