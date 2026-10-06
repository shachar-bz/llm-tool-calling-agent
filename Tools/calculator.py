import ast
import math

SUPPORTED_CALCULATOR_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

SUPPORTED_CALCULATOR_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left ** right,
}

SUPPORTED_CALCULATOR_UNARY_OPERATORS = {
    ast.USub: lambda value: -value,
}

CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression and return the numeric result as a string. "
            "Use this math computation such as multiplication, division, exponentiation, "
            "square roots, percentages, and final rounding. Supports +, -, *, /, **, "
            "parentheses, sqrt, log, abs, round, min, and max. Never compute math yourself, always use this tool."
        ),
        "parameters": {
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
    },
}


def evaluate_calculator_node(node):
    if isinstance(node, ast.Expression):
        return evaluate_calculator_node(node.body)

    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value
        raise ValueError("unsafe expression")

    if isinstance(node, ast.BinOp):
        operator = SUPPORTED_CALCULATOR_BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError("unsafe expression")
        left = evaluate_calculator_node(node.left)
        right = evaluate_calculator_node(node.right)
        return operator(left, right)

    if isinstance(node, ast.UnaryOp):
        operator = SUPPORTED_CALCULATOR_UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError("unsafe expression")
        value = evaluate_calculator_node(node.operand)
        return operator(value)

    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name):
            raise ValueError("unsafe expression")
        function = SUPPORTED_CALCULATOR_FUNCTIONS.get(node.func.id)
        if function is None:
            raise ValueError("unsafe expression")
        arguments = [evaluate_calculator_node(argument) for argument in node.args]
        return function(*arguments)

    raise ValueError("unsafe expression")


def calculator(expression: str) -> str:
    try:
        parsed_expression = ast.parse(expression, mode="eval")
        result = evaluate_calculator_node(parsed_expression)
        return str(result)
    except Exception as exception:
        return str(exception)
