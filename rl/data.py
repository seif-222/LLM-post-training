"""
GSM8K loading and prompt building.

Each problem is {"question": str, "answer": float, "solution": str}.
`solution` is the reference worked answer (ends in "#### <n>"); RL never trains on it,
it's kept only for inspecting failures.
"""
import re
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


def extract_final_answer(solution_text):
    """GSM8K solutions end with '#### <number>' -- pull that number out as a float."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", solution_text)
    assert match, f"couldn't find '#### <number>' in: {solution_text!r}"
    return float(match.group(1).replace(',', ''))


def load_gsm8k(split):
    """split: 'train' (7,473) or 'test' (1,319)."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": r["question"], "answer": extract_final_answer(r["answer"]), "solution": r["answer"]}
            for r in ds]


def build_prompt(tokenizer, question, style="hash"):
    """Chat-formatted prompt string ending at the start of the assistant turn."""
    messages = [{"role": "user", "content": PROMPTS[style].format(question=question)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
