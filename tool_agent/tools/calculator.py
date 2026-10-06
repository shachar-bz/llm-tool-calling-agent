"""Safe arithmetic: evaluates an expression by walking its AST, never with eval()."""

import ast
import math
import operator

from .base import Tool

FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.USub: operator.neg,
}


def calculator(expression: str) -> str:
    return str(_evaluate(ast.parse(expression, mode="eval").body))


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        return BINARY_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        return FUNCTIONS[node.func.id](*(_evaluate(argument) for argument in node.args))

    raise ValueError(f"unsupported expression: {ast.unparse(node)}")


CALCULATOR = Tool(
    name="calculator",
    description=(
        "Evaluate an arithmetic expression and return the numeric result as a string. "
        "Use this math computation such as multiplication, division, exponentiation, "
        "square roots, percentages, and final rounding. Supports +, -, *, /, **, "
        "parentheses, sqrt, log, abs, round, min, and max. Never compute math yourself, always use this tool."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "A Python arithmetic expression, e.g. '(45.20 + 12.50) / 100' "
                    "or 'round(sqrt(2300), 4)'."
                ),
            },
        },
        "required": ["expression"],
    },
    handler=calculator,
)
