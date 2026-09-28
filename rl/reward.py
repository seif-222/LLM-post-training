"""
Answer extraction and reward for GSM8K.

parse_answer looks for, in order of preference:
  1. a formatted answer -- '#### <n>' or '\\boxed{<n>}', whichever appears last in the text
  2. otherwise the last number anywhere ("last" fallback)
and reports which path it used, so eval can separate "wrong" from "right but unformatted".

Reward is 0/1 on correctness. "strict" only accepts formatted answers (the default for RL, so
formatting is implicitly rewarded); "lenient" also accepts the last-number fallback.
"""
import math
import re

# A number: either comma-grouped in threes (1,080 / 12,345.5) or plain digits (1080 / 3.50).
# Comma rule stops lists like "1,2,3" from being read as 123.
_NUM = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
NUM_RE = re.compile(_NUM)
FRAC_RE = re.compile(r"\\d?frac\{\s*(" + _NUM + r")\s*\}\{\s*(" + _NUM + r")\s*\}")
SLASH_RE = re.compile(r"(" + _NUM + r")\s*/\s*(" + _NUM + r")")
HASH_RE = re.compile(r"####([^\n]*)")


def _to_float(s):
    return float(s.replace(",", ""))


def _clean(text):
    # unicode minus and LaTeX dollar signs would otherwise break the number regex
    return text.replace("\u2212", "-").replace("\\$", "$").replace("\\!", "")


def number_in(segment):
    """First number in a short answer segment. Handles \\frac{a}{b} and a/b."""
    segment = _clean(segment)
    for pat in (FRAC_RE, SLASH_RE):
        m = pat.search(segment)
        if m:
            den = _to_float(m.group(2))
            if den != 0:
                return _to_float(m.group(1)) / den
    m = NUM_RE.search(segment)
    return _to_float(m.group(0)) if m else None


def _last_boxed(text):
    """(start index, content) of the last \\boxed{...}, brace-matched. None if absent/unclosed."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, text[i:j]
    return None


def parse_answer(text):
    """-> (value: float | None, source: 'hash' | 'boxed' | 'last' | None)."""
    candidates = []                                           # (position, value, source)
    hashes = list(HASH_RE.finditer(text))
    if hashes:
        m = hashes[-1]
        candidates.append((m.start(), number_in(m.group(1)), "hash"))
    boxed = _last_boxed(text)
    if boxed:
        candidates.append((boxed[0], number_in(boxed[1]), "boxed"))
    candidates = [c for c in candidates if c[1] is not None]
    if candidates:
        _, value, source = max(candidates)                     # the one written last wins
        return value, source

    nums = NUM_RE.findall(_clean(text))
    if nums:
        return _to_float(nums[-1]), "last"
    return None, None


def is_correct(pred, gold):
    return pred is not None and math.isclose(pred, gold, rel_tol=0, abs_tol=1e-4)


def score(text, gold):
    """Everything eval wants to know about one completion."""
    pred, source = parse_answer(text)
    correct = is_correct(pred, gold)
    return {"pred": pred, "source": source,
            "correct_strict": correct and source in ("hash", "boxed"),
            "correct_lenient": correct}


def reward(text, gold, mode="strict", format_weight=0.0):
    """RL reward: 1 for a correct answer (strict: must be formatted), plus an optional
    format_weight bonus for having a formatted answer at all, right or wrong."""
    s = score(text, gold)
    r = float(s["correct_strict"] if mode == "strict" else s["correct_lenient"])
    if format_weight and s["source"] in ("hash", "boxed"):
        r += format_weight
    return r
