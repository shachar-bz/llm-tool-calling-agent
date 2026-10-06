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


@pytest.mark.parametrize("expression", ["9 ** 9 ** 9", "(2 ** 1000) ** 1000", "10 ** 5000"])
def test_refuses_huge_powers_before_computing_them(expression):
    with pytest.raises(ValueError, match="result too large"):
        calculator(expression)


def test_allows_ordinary_powers():
    assert calculator("2 ** 64") == "18446744073709551616"
    assert calculator("1.05 ** 10") == str(1.05 ** 10)
    assert calculator("2 ** -1") == "0.5"
