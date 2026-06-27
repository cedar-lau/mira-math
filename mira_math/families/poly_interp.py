"""Polynomial interpolation with missing point."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random

from mira_math.utils.poly import lagrange_eval


FAMILY_NAME = "poly_interpolation"
FAMILY_TYPE = "B"  # Variable hint slot: any of the n+1 interpolation points


def _eval_poly(coeffs: List[int], x: int) -> int:
    total = 0
    power = 1
    for c in coeffs:
        total += c * power
        power *= x
    return total


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    degree = 2 + min(difficulty, 2)
    coeffs = [rng.randint(-3 - difficulty, 3 + difficulty) for _ in range(degree + 1)]

    xs = list(range(0, degree + 2 + difficulty))
    rng.shuffle(xs)
    points_x = xs[: degree + 1]
    points = [(x, _eval_poly(coeffs, x)) for x in points_x]

    target_candidates = [x for x in xs if x not in points_x]
    if not target_candidates:
        target_candidates = [points_x[-1] + 1]
    target_x = target_candidates[0]
    target_y = _eval_poly(coeffs, target_x)

    agent_a = points[:-1]
    agent_b = points[-1:]

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "int",
        "global_solution": {"value": int(target_y)},
        "global_metadata": {
            "degree": degree,
            "target_x": target_x,
            "coeffs": coeffs,
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": ["p"],
                    "externals": [],
                    "constraints_text": [f"p({x}) = {y}" for x, y in agent_a],
                    "constraints_machine": [
                        {"type": "point", "x": x, "y": y} for x, y in agent_a
                    ],
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": ["p"],
                    "externals": [],
                    "constraints_text": [f"p({x}) = {y}" for x, y in agent_b],
                    "constraints_machine": [
                        {"type": "point", "x": x, "y": y} for x, y in agent_b
                    ],
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "point",
                    "x": agent_b[0][0],
                    "y": agent_b[0][1],
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }
    return instance


def _extract_points(view: Dict[str, Any]) -> List[Tuple[int, int]]:
    points = []
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] == "point":
            points.append((int(c["x"]), int(c["y"])))
    return points


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    degree = int(instance["global_metadata"]["degree"])
    points = _extract_points(agent_view)
    return len(points) <= degree


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "point":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "point", "x": hint["x"], "y": hint["y"]}
        )
        new_view["private_data"]["constraints_text"].append(f"p({hint['x']}) = {hint['y']}")
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    degree = int(instance["global_metadata"]["degree"])
    points = _extract_points(agent_view)
    return len(points) >= degree + 1


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    points: List[Tuple[int, int]] = []
    for v in instance["agent_views"]:
        points.extend(_extract_points(v))
    degree = int(instance["global_metadata"]["degree"])
    target_x = int(instance["global_metadata"]["target_x"])
    if len(points) < degree + 1:
        raise ValueError("not enough points to solve polynomial")
    value = lagrange_eval(points[: degree + 1], target_x)
    if value.denominator != 1:
        raise ValueError("non-integer polynomial value")
    return {"value": int(value)}

