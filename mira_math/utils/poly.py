"""Polynomial helpers for exact interpolation."""
from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple


def lagrange_eval(points: List[Tuple[int, int]], x: int) -> Fraction:
    """Evaluate interpolating polynomial at x using Lagrange formula."""
    xq = Fraction(x)
    total = Fraction(0)
    for i, (xi, yi) in enumerate(points):
        xi_f = Fraction(xi)
        yi_f = Fraction(yi)
        num = Fraction(1)
        den = Fraction(1)
        for j, (xj, _) in enumerate(points):
            if i == j:
                continue
            xj_f = Fraction(xj)
            num *= (xq - xj_f)
            den *= (xi_f - xj_f)
        total += yi_f * (num / den)
    return total

