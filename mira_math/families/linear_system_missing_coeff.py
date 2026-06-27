"""Linear system Ax = b with one matrix coefficient masked.

Agent A holds: every coefficient of A and every RHS value b[i], EXCEPT one
  entry A[missing_row][missing_col] is absent from A's constraints.
Agent B holds: exactly that one entry A[missing_row][missing_col].

A must scan all (row, col) coefficient sentences and notice which (i,j) entry
is missing, then ask B for it. After receiving the hint, A has the full
n x n system and solves via Gaussian elimination, reporting x[target_k].

Answer: x[target_k] as a fully reduced fraction {numerator, denominator}.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List
import random

FAMILY_NAME = "linear_system_missing_coeff"
FAMILY_TYPE = "B"  # Variable hint slot: any matrix coefficient A[missing_row][missing_col]

_COEFF_TEMPLATES = [
    "In equation {i}, the coefficient of x{j} is {v}.",
    "Equation {i} has coefficient {v} for variable x{j}.",
    "The multiplier of x{j} in equation {i} equals {v}.",
    "Row {i} of the coefficient matrix has {v} in column {j}.",
    "The entry A[{i}][{j}] of the system matrix is {v}.",
]

_RHS_TEMPLATES = [
    "The right-hand side of equation {i} is {b}.",
    "Equation {i} has constant term {b}.",
    "The value of b[{i}] is {b}.",
    "Equation {i} evaluates to {b} on the right-hand side.",
]


def _nl_coeff(rng: random.Random, i: int, j: int, v: int) -> str:
    return rng.choice(_COEFF_TEMPLATES).format(i=i, j=j, v=v)


def _nl_rhs(rng: random.Random, i: int, b: int) -> str:
    return rng.choice(_RHS_TEMPLATES).format(i=i, b=b)


def _gauss_solve(A_mat: List[List[Fraction]], b_vec: List[Fraction], n: int) -> List[Fraction]:
    mat = [list(A_mat[i]) + [b_vec[i]] for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if mat[r][col] != 0), None)
        if pivot is None:
            raise ValueError("Singular matrix")
        mat[col], mat[pivot] = mat[pivot], mat[col]
        p = mat[col][col]
        mat[col] = [x / p for x in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [mat[r][k] - f * mat[col][k] for k in range(n + 1)]
    return [mat[i][n] for i in range(n)]


def _det_nonzero(A_mat: List[List[Fraction]], n: int) -> bool:
    mat = [list(A_mat[i]) for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if mat[r][col] != 0), None)
        if pivot is None:
            return False
        mat[col], mat[pivot] = mat[pivot], mat[col]
        p = mat[col][col]
        mat[col] = [x / p for x in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [mat[r][k] - f * mat[col][k] for k in range(n)]
    return True


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    n = difficulty + 1          # 2, 3, 4
    max_coeff = 4 + 3 * (difficulty - 1)   # 4, 7, 10

    while True:
        A = [[rng.randint(-max_coeff, max_coeff) for _ in range(n)] for _ in range(n)]
        # Ensure no zero rows and non-singular
        A_frac = [[Fraction(A[i][j]) for j in range(n)] for i in range(n)]
        if not _det_nonzero(A_frac, n):
            continue

        b = [rng.randint(-8, 8) for _ in range(n)]
        b_frac = [Fraction(b[i]) for i in range(n)]

        try:
            x = _gauss_solve(A_frac, b_frac, n)
        except ValueError:
            continue

        missing_row = rng.randint(0, n - 1)
        missing_col = rng.randint(0, n - 1)
        target_k = rng.randint(0, n - 1)

        answer = x[target_k]
        if answer.denominator > 300:
            continue

        break

    # Agent A: all coefficients except (missing_row, missing_col) + all RHS
    a_text: List[str] = []
    a_machine: List[Dict[str, Any]] = []

    for i in range(n):
        for j in range(n):
            if i == missing_row and j == missing_col:
                continue
            v = A[i][j]
            a_text.append(_nl_coeff(rng, i, j, v))
            a_machine.append({"type": "coeff", "row": i, "col": j, "value": v})

    for i in range(n):
        a_text.append(_nl_rhs(rng, i, b[i]))
        a_machine.append({"type": "rhs", "row": i, "value": b[i]})

    # Agent B: just the missing coefficient
    hint_val = A[missing_row][missing_col]
    b_text = [f"A[{missing_row}][{missing_col}] = {hint_val}"]
    b_machine = [{"type": "coeff", "row": missing_row, "col": missing_col, "value": hint_val}]

    return {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "rational",
        "global_solution": {
            "numerator": int(answer.numerator),
            "denominator": int(answer.denominator),
        },
        "global_metadata": {
            "n": n,
            "missing_row": missing_row,
            "missing_col": missing_col,
            "target_k": target_k,
            "A": A,
            "b": b,
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": [f"x{target_k}"],
                    "externals": [],
                    "constraints_text": a_text,
                    "constraints_machine": a_machine,
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": [f"x{target_k}"],
                    "externals": [],
                    "constraints_text": b_text,
                    "constraints_machine": b_machine,
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "coeff",
                    "row": missing_row,
                    "col": missing_col,
                    "value": hint_val,
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }


def _known_coeffs(view: Dict[str, Any]) -> set:
    return {
        (c["row"], c["col"])
        for c in view["private_data"]["constraints_machine"]
        if c["type"] == "coeff"
    }


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n"]
    known = _known_coeffs(agent_view)
    return not all((i, j) in known for i in range(n) for j in range(n))


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "coeff":
        r, c, v = hint["row"], hint["col"], hint["value"]
        new_view["private_data"]["constraints_machine"].append(
            {"type": "coeff", "row": r, "col": c, "value": v}
        )
        new_view["private_data"]["constraints_text"].append(f"A[{r}][{c}] = {v}")
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n"]
    known = _known_coeffs(agent_view)
    return all((i, j) in known for i in range(n) for j in range(n))


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    n = instance["global_metadata"]["n"]
    A = instance["global_metadata"]["A"]
    b = instance["global_metadata"]["b"]
    target_k = instance["global_metadata"]["target_k"]
    A_frac = [[Fraction(A[i][j]) for j in range(n)] for i in range(n)]
    b_frac = [Fraction(b[i]) for i in range(n)]
    x = _gauss_solve(A_frac, b_frac, n)
    result = x[target_k]
    return {"numerator": int(result.numerator), "denominator": int(result.denominator)}
