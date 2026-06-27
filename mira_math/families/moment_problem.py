"""Moment problem on support {0,1,2} with rational probabilities."""
from __future__ import annotations

from typing import Any, Dict, List
import random

from mira_math.utils.exact import solve_linear_system


FAMILY_NAME = "moment_problem"
FAMILY_TYPE = "A"  # Fixed hint slot: always the n2 moment equation


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
    return A, b


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    variables = ["q0", "q1", "q2"]
    denom = 6 + 2 * difficulty

    while True:
        q0 = rng.randint(1, denom - 2)
        q1 = rng.randint(1, denom - q0 - 1)
        q2 = denom - q0 - q1
        if q2 <= 0:
            continue

        m1 = q1 + 2 * q2
        m2 = q1 + 4 * q2

        eq_sum = {"type": "lin_eq", "coeffs": {"q0": 1, "q1": 1, "q2": 1}, "rhs": denom}
        eq_m1 = {"type": "lin_eq", "coeffs": {"q0": 0, "q1": 1, "q2": 2}, "rhs": m1}
        eq_m2 = {"type": "lin_eq", "coeffs": {"q0": 0, "q1": 1, "q2": 4}, "rhs": m2}

        # Check uniqueness with all constraints
        A_full = [
            [1, 1, 1],
            [0, 1, 2],
            [0, 1, 4],
        ]
        b_full = [denom, m1, m2]
        if solve_linear_system(A_full, b_full)["status"] != "unique":
            continue

        instance = {
            "id": instance_id,
            "family": FAMILY_NAME,
            "difficulty": difficulty,
            "n_agents": 2,
            "answer_type": "dict_int",
            "global_solution": {"q0": q0, "q1": q1, "q2": q2},
            "global_metadata": {
                "support": [0, 1, 2],
                "denominator": denom,
                "m1": m1,
                "m2": m2,
            },
            "agent_views": [
                {
                    "agent_id": "A",
                    "private_data": {
                        "unknowns": variables,
                        "externals": [],
                        "constraints_text": [
                            _format_lin_eq([1, 1, 1], variables, denom),
                            _format_lin_eq([0, 1, 2], variables, m1),
                        ],
                        "constraints_machine": [eq_sum, eq_m1],
                    },
                },
                {
                    "agent_id": "B",
                    "private_data": {
                        "unknowns": variables,
                        "externals": [],
                        "constraints_text": [
                            _format_lin_eq([1, 1, 1], variables, denom),
                            _format_lin_eq([0, 1, 4], variables, m2),
                        ],
                        "constraints_machine": [eq_sum, eq_m2],
                    },
                },
            ],
            "minimal_hint_spec": {
                "k_min": 1,
                "atomic_hints": [
                    {
                        "hint_id": "h1",
                        "kind": "lin_eq",
                        "coeffs": eq_m2["coeffs"],
                        "rhs": eq_m2["rhs"],
                        "providers": ["B"],
                        "consumers": ["A"],
                    }
                ],
            },
        }
        return instance


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    variables = ["q0", "q1", "q2"]
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
    variables = ["q0", "q1", "q2"]
    A, b = _build_linear_system(agent_view, variables)
    result = solve_linear_system(A, b)
    return result["status"] == "unique"


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    variables = ["q0", "q1", "q2"]
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
        raise ValueError("moment system not uniquely solvable")
    sol = result["solution"]
    return {variables[i]: int(sol[i]) for i in range(len(variables))}
