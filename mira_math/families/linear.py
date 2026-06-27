"""Rank-deficient linear systems with separator variable."""
from __future__ import annotations

from typing import Any, Dict, List
import random

from mira_math.utils.exact import solve_linear_system


FAMILY_NAME = "linear_system_separator"
FAMILY_TYPE = "A"  # Fixed hint slot: always the separator variable c


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


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    variables = ["a", "b", "c"]
    coef_low, coef_high = -3 - difficulty, 3 + difficulty

    # Pick a global solution
    sol = {v: rng.randint(-5 - difficulty, 5 + difficulty) for v in variables}

    # Build two independent equations in 3 vars such that fixing c still leaves
    # a unique solution for (a, b).  This requires the rows restricted to the
    # (a, b) sub-space to also have rank 2.
    sep_idx = variables.index("c")
    other_idx = [i for i in range(len(variables)) if i != sep_idx]
    while True:
        row1 = [_random_nonzero_int(rng, coef_low, coef_high) for _ in variables]
        row2 = [_random_nonzero_int(rng, coef_low, coef_high) for _ in variables]
        if _rank_of_rows([row1, row2]) != 2:
            continue
        # After substituting c, the (a,b) sub-system must be uniquely solvable.
        rows_ab = [[row1[i] for i in other_idx], [row2[i] for i in other_idx]]
        if _rank_of_rows(rows_ab) == 2:
            break

    b1 = sum(row1[i] * sol[variables[i]] for i in range(3))
    b2 = sum(row2[i] * sol[variables[i]] for i in range(3))

    eq1 = {"type": "lin_eq", "coeffs": {variables[i]: row1[i] for i in range(3)}, "rhs": b1}
    eq2 = {"type": "lin_eq", "coeffs": {variables[i]: row2[i] for i in range(3)}, "rhs": b2}
    eq3 = {"type": "value", "var": "c", "value": sol["c"]}

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
                    "constraints_text": [f"c = {sol['c']}"],
                    "constraints_machine": [eq3],
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "value",
                    "var": "c",
                    "value": sol["c"],
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }
    return instance


def _format_lin_eq(row: List[int], variables: List[str], rhs: int) -> str:
    terms = []
    for coef, var in zip(row, variables):
        if coef == 0:
            continue
        if coef == 1:
            terms.append(var)
        elif coef == -1:
            terms.append(f"-{var}")
        else:
            terms.append(f"{coef}{var}")
    lhs = " + ".join(terms).replace("+-", "-")
    return f"{lhs} = {rhs}"


def _build_linear_system(view: Dict[str, Any], variables: List[str]) -> List[List[int]]:
    A: List[List[int]] = []
    b: List[int] = []
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] == "lin_eq":
            row = [int(c["coeffs"].get(v, 0)) for v in variables]
            A.append(row)
            b.append(int(c["rhs"]))
        elif c["type"] == "value":
            row = [0 for _ in variables]
            row[variables.index(c["var"])] = 1
            A.append(row)
            b.append(int(c["value"]))
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
    if hint["kind"] == "value":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "value", "var": hint["var"], "value": hint["value"]}
        )
        new_view["private_data"]["constraints_text"].append(f"{hint['var']} = {hint['value']}")
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    variables = instance["global_metadata"]["variables"]
    A, b = _build_linear_system(agent_view, variables)
    result = solve_linear_system(A, b)
    return result["status"] == "unique"


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    # Global constraints are the union of all agent constraints
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
        elif c["type"] == "value":
            row = [0 for _ in variables]
            row[variables.index(c["var"])] = 1
            A.append(row)
            b.append(int(c["value"]))

    result = solve_linear_system(A, b)
    if result["status"] != "unique":
        raise ValueError("global system not uniquely solvable")
    sol = result["solution"]
    return {variables[i]: int(sol[i]) for i in range(len(variables))}

