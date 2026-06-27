"""Triangle graph edge reconstruction from path sums."""
from __future__ import annotations

from typing import Any, Dict, List
import random

from mira_math.utils.exact import solve_linear_system


FAMILY_NAME = "graph_path_sums"
FAMILY_TYPE = "A"  # Fixed hint slot: always eq3 (the e01+e20 path sum)


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
    variables = ["e01", "e12", "e20"]
    edge_range = 2 + difficulty

    e01 = rng.randint(1, edge_range)
    e12 = rng.randint(1, edge_range)
    e20 = rng.randint(1, edge_range)

    s1 = e01 + e12
    s2 = e12 + e20
    s3 = e01 + e20

    eq1 = {"type": "lin_eq", "coeffs": {"e01": 1, "e12": 1, "e20": 0}, "rhs": s1}
    eq2 = {"type": "lin_eq", "coeffs": {"e01": 0, "e12": 1, "e20": 1}, "rhs": s2}
    eq3 = {"type": "lin_eq", "coeffs": {"e01": 1, "e12": 0, "e20": 1}, "rhs": s3}

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "dict_int",
        "global_solution": {"e01": e01, "e12": e12, "e20": e20},
        "global_metadata": {
            "variables": variables,
            "paths": ["01-12", "12-20", "01-20"],
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": variables,
                    "externals": [],
                    "constraints_text": [
                        _format_lin_eq([1, 1, 0], variables, s1),
                        _format_lin_eq([0, 1, 1], variables, s2),
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
                        _format_lin_eq([0, 1, 1], variables, s2),
                        _format_lin_eq([1, 0, 1], variables, s3),
                    ],
                    "constraints_machine": [eq2, eq3],
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
        raise ValueError("graph system not uniquely solvable")
    sol = result["solution"]
    return {variables[i]: int(sol[i]) for i in range(len(variables))}
