import pytest

from tool_agent.tools.calculator import calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", "7"),
        ("(45.25 + 12.75) / 100", "0.58"),
        ("-2 ** 2", "-4"),
        ("round(87.45 / 1240.3 * 100, 2)", "7.05"),
        ("round(sqrt(2300), 4)", "47.9583"),
        ("max(1, abs(-5), min(3, 4))", "5"),
    ],
)
def test_evaluates_arithmetic(expression, expected):
    assert calculator(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('secrets.txt')",
        "(1).__class__",
        "'a' * 3",
        "x + 1",
        "round(1.5, ndigits=0)",
    ],
)
def test_rejects_anything_but_arithmetic(expression):
    with pytest.raises(ValueError, match="unsupported expression"):
        calculator(expression)
