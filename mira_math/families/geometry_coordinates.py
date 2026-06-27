"""Coordinate geometry with missing line constraint."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random
import math

from mira_math.utils.exact import solve_linear_system


FAMILY_NAME = "geometry_coordinates"
FAMILY_TYPE = "A"  # Fixed hint slot: always B's line equation


def _format_lin_eq(row: List[int], variables: List[str], rhs: int) -> str:
    """
    Format a linear equation like: x - 2y = 3

    CHANGE: Rewrote formatting to avoid producing "x + -2y".
    """
    parts: List[str] = []
    for coef, var in zip(row, variables):
        if coef == 0:
            continue

        abs_coef = abs(coef)
        if abs_coef == 1:
            term = f"{var}"
        else:
            term = f"{abs_coef}{var}"

        if not parts:
            # first term keeps leading '-' if needed, no leading '+'
            parts.append(f"-{term}" if coef < 0 else term)
        else:
            parts.append(f"- {term}" if coef < 0 else f"+ {term}")

    lhs = " ".join(parts) if parts else "0"
    return f"{lhs} = {rhs}"


def _build_linear_system(view: Dict[str, Any], variables: List[str]) -> Tuple[List[List[int]], List[int]]:
    """
    Extract A, b from the agent view.

    CHANGE: return type annotation corrected (it returns (A, b)).
    """
    A: List[List[int]] = []
    b: List[int] = []
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] == "lin_eq":
            row = [int(c["coeffs"].get(v, 0)) for v in variables]
            A.append(row)
            b.append(int(c["rhs"]))
    return A, b


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    variables = ["x", "y"]
    coef_low, coef_high = -3 - difficulty, 3 + difficulty
    x = rng.randint(-4 - difficulty, 4 + difficulty)
    y = rng.randint(-4 - difficulty, 4 + difficulty)

    # Build two non-parallel lines through (x, y)
    while True:
        a1 = rng.randint(coef_low, coef_high)
        b1 = rng.randint(coef_low, coef_high)
        a2 = rng.randint(coef_low, coef_high)
        b2 = rng.randint(coef_low, coef_high)
        if a1 == 0 and b1 == 0:
            continue
        if a2 == 0 and b2 == 0:
            continue
        if a1 * b2 - a2 * b1 == 0:
            continue
        break

    c1 = a1 * x + b1 * y
    c2 = a2 * x + b2 * y

    # Choose a point Q at an integer distance from (x, y)
    dx, dy = rng.choice([(3, 4), (4, 3), (5, 12), (12, 5)])
    qx = x + dx
    qy = y + dy
    dist_sq = dx * dx + dy * dy
    dist = math.isqrt(dist_sq)  # CHANGE: exact integer sqrt
    if dist * dist != dist_sq:
        raise ValueError("chosen (dx, dy) did not yield an integer distance (unexpected)")

    eq1 = {"type": "lin_eq", "coeffs": {"x": a1, "y": b1}, "rhs": c1}
    eq2 = {"type": "lin_eq", "coeffs": {"x": a2, "y": b2}, "rhs": c2}

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "dict_int",  # CHANGE: was "int" but solutions are {"value": ...}
        "global_solution": {"value": dist},
        "global_metadata": {
            "variables": variables,
            "point_q": [qx, qy],
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": variables,
                    "externals": [],
                    "constraints_text": [
                        _format_lin_eq([a1, b1], variables, c1),
                        f"Q = ({qx}, {qy})",
                    ],
                    "constraints_machine": [eq1],
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": variables,
                    "externals": [],
                    "constraints_text": [
                        _format_lin_eq([a2, b2], variables, c2),
                        f"Q = ({qx}, {qy})",
                    ],
                    "constraints_machine": [eq2],
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "lin_eq",
                    "coeffs": eq2["coeffs"],
                    "rhs": eq2["rhs"],
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

        # CHANGE: keep stable variable ordering using agent's unknown list (not dict key order)
        variables = list(new_view["private_data"].get("unknowns", [])) or list(hint["coeffs"].keys())
        row = [int(hint["coeffs"].get(v, 0)) for v in variables]

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
        raise ValueError("geometry system not uniquely solvable")

    sol = result["solution"]
    x = int(sol[0])
    y = int(sol[1])

    qx, qy = instance["global_metadata"]["point_q"]
    dx = x - int(qx)
    dy = y - int(qy)
    dist_sq = dx * dx + dy * dy

    dist = math.isqrt(dist_sq)  # CHANGE: exact integer sqrt
    if dist * dist != dist_sq:
        raise ValueError("distance not integer")

    return {"value": dist}
