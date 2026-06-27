"""Rank-deficient linear systems with shared variable."""
from __future__ import annotations

from typing import Any, Dict, List
import random

from mira_math.utils.exact import solve_linear_system


FAMILY_NAME = "rankdef_linear_shared"
FAMILY_TYPE = "A"  # Fixed hint slot: always eq3 (one fixed equation from B's two)


def _random_nonzero_int(rng: random.Random, low: int, high: int) -> int:
    while True:
        v = rng.randint(low, high)
        if v != 0:
            return v


def _rank_of_rows(rows: List[List[int]]) -> int:
    if not rows:
        return 0
    b = [0 for _ in rows]
    res = solve_linear_system(rows, b)
    return res["rank"]


def _format_lin_eq(row: List[int], variables: List[str], rhs: int) -> str:
    parts: List[tuple[str, str]] = []

    for coef, var in zip(row, variables):
        if coef == 0:
            continue
        sign = "-" if coef < 0 else "+"
        a = abs(coef)
        if a == 1:
            term = f"{var}"
        else:
            term = f"{a}{var}"
        parts.append((sign, term))

    if not parts:
        lhs = "0"
    else:
        first_sign, first_term = parts[0]
        lhs = (f"-{first_term}" if first_sign == "-" else first_term)
        for sign, term in parts[1:]:
            lhs += f" {sign} {term}"

    return f"{lhs} = {rhs}"


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    variables = ["x", "y", "z"]
    coef_low, coef_high = -3 - difficulty, 3 + difficulty

    sol = {v: rng.randint(-5 - difficulty, 5 + difficulty) for v in variables}

    # Agent A rows (rank 2)
    while True:
        row1 = [_random_nonzero_int(rng, coef_low, coef_high) for _ in variables]
        row2 = [_random_nonzero_int(rng, coef_low, coef_high) for _ in variables]
        if _rank_of_rows([row1, row2]) == 2:
            break

    # Agent B rows (rank 2), with one row independent of A's rows
    while True:
        row3 = [_random_nonzero_int(rng, coef_low, coef_high) for _ in variables]
        if _rank_of_rows([row1, row2, row3]) == 3:
            break

    while True:
        row4 = [_random_nonzero_int(rng, coef_low, coef_high) for _ in variables]
        # row4 must be independent of row3 AND must also complete A's system on its own,
        # so that whichever of B's equations A receives is sufficient.
        if _rank_of_rows([row3, row4]) == 2 and _rank_of_rows([row1, row2, row4]) == 3:
            break

    b1 = sum(row1[i] * sol[variables[i]] for i in range(3))
    b2 = sum(row2[i] * sol[variables[i]] for i in range(3))
    b3 = sum(row3[i] * sol[variables[i]] for i in range(3))
    b4 = sum(row4[i] * sol[variables[i]] for i in range(3))

    eq1 = {"type": "lin_eq", "coeffs": {variables[i]: row1[i] for i in range(3)}, "rhs": b1}
    eq2 = {"type": "lin_eq", "coeffs": {variables[i]: row2[i] for i in range(3)}, "rhs": b2}
    eq3 = {"type": "lin_eq", "coeffs": {variables[i]: row3[i] for i in range(3)}, "rhs": b3}
    eq4 = {"type": "lin_eq", "coeffs": {variables[i]: row4[i] for i in range(3)}, "rhs": b4}

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "dict_int",
        "global_solution": sol,
        "global_metadata": {
            "variables": variables,
            "domains": {v: "Z" for v in variables},
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": variables,
                    "externals": [],
                    "constraints_text": [
                        _format_lin_eq(row1, variables, b1),
                        _format_lin_eq(row2, variables, b2),
                    ],
                    "constraints_machine": [eq1, eq2],
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": variables,
                    "externals": [],
                    "constraints_text": [
                        _format_lin_eq(row3, variables, b3),
                        _format_lin_eq(row4, variables, b4),
                    ],
                    "constraints_machine": [eq3, eq4],
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "lin_eq",
                    "coeffs": eq3["coeffs"],
                    "rhs": eq3["rhs"],
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }
    return instance


def _build_linear_system(view: Dict[str, Any], variables: List[str]) -> List[List[int]]:
    A: List[List[int]] = []
    b: List[int] = []
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] == "lin_eq":
            row = [int(c["coeffs"].get(v, 0)) for v in variables]
            A.append(row)
            b.append(int(c["rhs"]))
    return A, b


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    variables = instance["global_metadata"]["variables"]
    A, b = _build_linear_system(agent_view, variables)
    result = solve_linear_system(A, b)
    return result["status"] != "unique"


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "lin_eq":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "lin_eq", "coeffs": hint["coeffs"], "rhs": hint["rhs"]}
        )
        variables = list(hint["coeffs"].keys())
        row = [hint["coeffs"][v] for v in variables]
        new_view["private_data"]["constraints_text"].append(
            _format_lin_eq(row, variables, int(hint["rhs"]))
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    variables = instance["global_metadata"]["variables"]
    A, b = _build_linear_system(agent_view, variables)
    result = solve_linear_system(A, b)
    return result["status"] == "unique"


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    variables = instance["global_metadata"]["variables"]
    all_constraints = []
    for v in instance["agent_views"]:
        all_constraints.extend(v["private_data"]["constraints_machine"])

    A: List[List[int]] = []
    b: List[int] = []
    for c in all_constraints:
        if c["type"] == "lin_eq":
            row = [int(c["coeffs"].get(v, 0)) for v in variables]
            A.append(row)
            b.append(int(c["rhs"]))

    result = solve_linear_system(A, b)
    if result["status"] != "unique":
        raise ValueError("global system not uniquely solvable")
    sol = result["solution"]
    return {variables[i]: int(sol[i]) for i in range(len(variables))}
