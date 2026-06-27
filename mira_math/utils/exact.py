"""Exact arithmetic helpers."""
from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple, Dict, Any


def to_fraction_matrix(A: List[List[Any]], b: List[Any]) -> Tuple[List[List[Fraction]], List[Fraction]]:
    A2 = [[Fraction(x) for x in row] for row in A]
    b2 = [Fraction(x) for x in b]
    return A2, b2


def rref(A: List[List[Fraction]], b: List[Fraction]) -> Tuple[List[List[Fraction]], List[Fraction], List[int]]:
    rows = len(A)
    cols = len(A[0]) if rows else 0
    r = 0
    pivot_cols: List[int] = []

    for c in range(cols):
        if r >= rows:
            break
        # Find pivot row
        pivot = None
        for i in range(r, rows):
            if A[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        # Swap into position
        if pivot != r:
            A[r], A[pivot] = A[pivot], A[r]
            b[r], b[pivot] = b[pivot], b[r]

        # Normalize pivot row
        pivot_val = A[r][c]
        A[r] = [x / pivot_val for x in A[r]]
        b[r] = b[r] / pivot_val

        # Eliminate other rows
        for i in range(rows):
            if i == r:
                continue
            factor = A[i][c]
            if factor == 0:
                continue
            A[i] = [A[i][j] - factor * A[r][j] for j in range(cols)]
            b[i] = b[i] - factor * b[r]

        pivot_cols.append(c)
        r += 1

    return A, b, pivot_cols


def solve_linear_system(A: List[List[Any]], b: List[Any]) -> Dict[str, Any]:
    if not A:
        return {"status": "none", "solution": None, "nullity": 0, "rank": 0}

    A2, b2 = to_fraction_matrix(A, b)
    A_rref, b_rref, pivots = rref(A2, b2)
    rows = len(A_rref)
    cols = len(A_rref[0]) if rows else 0

    # Check consistency: row of zeros with nonzero RHS
    for i in range(rows):
        if all(A_rref[i][j] == 0 for j in range(cols)) and b_rref[i] != 0:
            return {"status": "none", "solution": None, "nullity": 0, "rank": len(pivots)}

    rank = len(pivots)
    nullity = cols - rank

    if nullity == 0:
        # Unique solution
        sol = [Fraction(0) for _ in range(cols)]
        for r, c in enumerate(pivots):
            sol[c] = b_rref[r]
        return {"status": "unique", "solution": sol, "nullity": 0, "rank": rank}

    return {"status": "infinite", "solution": None, "nullity": nullity, "rank": rank}

