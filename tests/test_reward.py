"""GSM8K answer parser and reward. Spec cases; do not loosen a failure to match a bug."""
import pytest

from rl.reward import number_in, parse_answer, reward, score


@pytest.mark.parametrize(
    "text, value, source",
    [
        ("so she has 18 left.\n#### 18", 18, "hash"),
        ("#### 1,080", 1080, "hash"),
        ("#### $1,080.50", 1080.5, "hash"),
        ("#### -7", -7, "hash"),
        ("#### 25%", 25, "hash"),
        ("#### 18.", 18, "hash"),
        ("The answer is \\boxed{42}.", 42, "boxed"),
        ("\\boxed{\\$18}", 18, "boxed"),
        ("\\boxed{18 \\text{ dollars}}", 18, "boxed"),
        ("\\boxed{\\frac{3}{4}}", 0.75, "boxed"),
        ("\\boxed{\\dfrac{1}{2}}", 0.5, "boxed"),
        ("#### 3/4", 0.75, "hash"),
        ("#### 5\nwait, actually \\boxed{6}", 6, "boxed"),
        ("\\boxed{6}\n#### 5", 5, "hash"),
        ("#### 10\n#### 12", 12, "hash"),
        ("She buys 3 apples then 5 more, so 8", 8, "last"),
        ("numbers 1,2,3", 3, "last"),
        ("costs $2,500 total", 2500, "last"),
        ("\u22125 degrees", -5, "last"),
        ("no numbers here", None, None),
        ("", None, None),
        ("#### \n the answer is 9", 9, "last"),
        ("\\boxed{ 7", 7, "last"),
    ],
)
def test_parse_answer(text, value, source):
    got_value, got_source = parse_answer(text)
    assert got_source == source
    if value is None:
        assert got_value is None
    else:
        assert got_value == pytest.approx(value)


@pytest.mark.parametrize(
    "text, gold, correct_strict, correct_lenient, pred, source",
    [
        ("#### 18", 18.0, True, True, 18.0, "hash"),
        ("so 18", 18.0, False, True, 18.0, "last"),
        ("#### 17", 18.0, False, False, 17.0, "hash"),
        ("#### 18.00001", 18.0, True, True, 18.00001, "hash"),
        ("#### 18.01", 18.0, False, False, 18.01, "hash"),
        ("", 18.0, False, False, None, None),
    ],
)
def test_score(text, gold, correct_strict, correct_lenient, pred, source):
    result = score(text, gold)
    assert result["correct_strict"] is correct_strict
    assert result["correct_lenient"] is correct_lenient
    assert result["source"] == source
    if pred is None:
        assert result["pred"] is None
    else:
        assert result["pred"] == pytest.approx(pred)


@pytest.mark.parametrize(
    "text, gold, kwargs, expected",
    [
        ("#### 18", 18, {}, 1.0),
        ("so 18", 18, {}, 0.0),
        ("so 18", 18, {"mode": "lenient"}, 1.0),
        ("#### 17", 18, {"format_weight": 0.1}, pytest.approx(0.1)),
        ("#### 18", 18, {"format_weight": 0.1}, pytest.approx(1.1)),
        ("so 17", 18, {"format_weight": 0.1}, 0.0),
    ],
)
def test_reward(text, gold, kwargs, expected):
    assert reward(text, gold, **kwargs) == expected


@pytest.mark.parametrize(
    "segment, expected",
    [
        ("", None),
        ("1/0", 1),
    ],
)
def test_number_in(segment, expected):
    got = number_in(segment)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)
