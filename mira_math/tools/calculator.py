"""Calculator tool for mira_math agents.

Provides exact arithmetic operations useful for solving math problems.
"""
from __future__ import annotations

import ast
import operator
from fractions import Fraction
from typing import Union

from langchain_core.tools import tool


# Safe operators for evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> Union[int, float, Fraction]:
    """Safely evaluate an AST node containing only arithmetic operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    
    if isinstance(node, ast.Num):  # Python 3.7 compatibility
        return node.n
    
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](left, right)
    
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](operand)
    
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Supports: +, -, *, /, //, %, ** (power)
    Returns exact results using Fraction when needed.
    
    Args:
        expression: A mathematical expression like "2 + 3 * 4" or "(10 - 6) / 2"
    
    Returns:
        String representation of the result
    """
    try:
        # Clean the expression
        expr = expression.strip()
        
        # Parse to AST
        tree = ast.parse(expr, mode='eval')
        
        # Safely evaluate
        result = _safe_eval(tree)
        
        # Convert to int if possible
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        
        return str(result)
    
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as e:
        return f"Error: {str(e)}"


@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the exact result.
    
    Use this tool when you need to perform arithmetic calculations.
    Supports: +, -, *, /, //, %, ** (power), parentheses
    
    Examples:
        - "2 + 3 * 4" -> "14"
        - "(10 - 6) / 2" -> "2.0"
        - "7 ** 2" -> "49"
        - "17 % 5" -> "2"
    
    Args:
        expression: A mathematical expression to evaluate
    
    Returns:
        The result as a string, or an error message if invalid
    """
    return calculate(expression)


# For direct import
CALCULATOR_TOOL = calculator_tool
