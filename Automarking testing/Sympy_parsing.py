"""
sympy_safe_parser_v2.py
-----------------------
Robust Sympy parser that accurately handles LaTeX (especially \frac and \ln).
"""

import re
import sympy
from sympy import Eq, simplify
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

def clean_input(expr: str) -> str:
    """
    Cleans LaTeX input safely for parsing.
    Does NOT replace braces in \frac.
    """
    if not expr or not isinstance(expr, str):
        return ""

    expr = expr.strip()

    # Remove formatting wrappers
    expr = re.sub(r"\\left|\\right", "", expr)
    expr = re.sub(r"\\,", "", expr)
    expr = re.sub(r"\\;", "", expr)
    expr = re.sub(r"\\!", "", expr)
    expr = re.sub(r"\\ ", "", expr)

    # Fix common functions
    replacements = {
        r"\\cdot": "*",
        r"\\times": "*",
        r"\\div": "/",
        r"\\ln": "log",
        r"\\log": "log",
        r"\\sin": "sin",
        r"\\cos": "cos",
        r"\\tan": "tan",
        r"\\sqrt": "sqrt",
        r"\\exp": "exp"
    }
    for k, v in replacements.items():
        expr = expr.replace(k, v)

    return expr


def safe_parse(expr_str: str):
    """
    Attempts to parse expression safely using latex2sympy2 or Sympy's parser.
    Returns Sympy object or None.
    """
    expr_str = clean_input(expr_str)

    # latex2sympy2 can handle most LaTeX directly
    try:
        parsed = latex2sympy(expr_str)
        return parsed
    except Exception as e1:
        try:
            parsed = parse_latex(expr_str)
            return parsed
        except Exception as e2:
            try:
                parsed = sympy.sympify(expr_str)
                return parsed
            except Exception as e3:
                print(f"[WARN] Could not parse: {expr_str}")
                print(f"  Errors: {e1}\n          {e2}\n          {e3}")
                return None


def test_parser():
    tests = [
        r"\frac{dy}{dx} = \frac{2x}{x^2 + 1} - \frac{1}{x}",
        r"y = \ln\left(\frac{x^2 + 1}{x}\right)",
        r"\int_0^1 x^2 \, dx",
        r"\sqrt{x^2 + 4x + 4}",
        r"\sin(x)^2 + \cos(x)^2 = 1",
        r"y = \exp(x) * \sin(x)",
        "(x^2 + 1)/(x + 1)"
    ]

    for expr in tests:
        print(f"\n🔹 Input: {expr}")
        parsed = safe_parse(expr)
        print(f"→ Parsed: {parsed}")
        print(f"→ Type: {type(parsed)}")


if __name__ == "__main__":
    test_parser()
