"""Piecewise linear function with missing branch threshold.

Agent A holds the query point x and all branch formulas, but not the threshold t.
Agent B holds x and the threshold t (so B knows which branch is active).
Neither can evaluate f(x) alone; together they determine the correct branch and compute the result.
Answer: f(x) as an integer.
"""
from __future__ import annotations

from typing import Any, Dict, List
import random


FAMILY_NAME = "piecewise_missing_threshold"
FAMILY_TYPE = "A"  # Fixed hint slot: always the threshold t


def _branch_text(slope: int, intercept: int) -> str:
    """Return a human-readable expression for slope*x + intercept."""
    parts: List[str] = []
    if slope == 1:
        parts.append("x")
    elif slope == -1:
        parts.append("-x")
    elif slope != 0:
        parts.append(f"{slope}x")

    if intercept > 0:
        parts.append(f"+ {intercept}" if parts else str(intercept))
    elif intercept < 0:
        parts.append(f"- {abs(intercept)}")
    elif not parts:
        parts.append("0")

    return " ".join(parts)


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    coeff_range = 2 + difficulty
    x_range = 5 + 3 * difficulty

    while True:
        t = rng.randint(-x_range // 2, x_range // 2)
        x = rng.randint(-x_range, x_range)
        if x == t:
            continue

        slope0 = rng.randint(-coeff_range, coeff_range)
        intercept0 = rng.randint(-coeff_range * 2, coeff_range * 2)
        slope1 = rng.randint(-coeff_range, coeff_range)
        intercept1 = rng.randint(-coeff_range * 2, coeff_range * 2)

        val0 = slope0 * x + intercept0
        val1 = slope1 * x + intercept1

        # The two branches must give distinct values at x,
        # otherwise knowing the threshold doesn't change the answer.
        if val0 == val1:
            continue

        break

    active_branch = 0 if x < t else 1
    answer = val0 if active_branch == 0 else val1

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "int",
        "global_solution": {"value": int(answer)},
        "global_metadata": {
            "x": x,
            "threshold": t,
            "active_branch": active_branch,
            "branches": [
                {"slope": slope0, "intercept": intercept0},
                {"slope": slope1, "intercept": intercept1},
            ],
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": ["f_x"],
                    "externals": [],
                    "constraints_text": [
                        f"x = {x}",
                        f"f(x) = {_branch_text(slope0, intercept0)}  if x < t",
                        f"f(x) = {_branch_text(slope1, intercept1)}  if x >= t",
                    ],
                    "constraints_machine": [
                        {"type": "query_point", "x": x},
                        {"type": "branch_formula", "branch": 0,
                         "slope": slope0, "intercept": intercept0, "condition": "lt"},
                        {"type": "branch_formula", "branch": 1,
                         "slope": slope1, "intercept": intercept1, "condition": "geq"},
                    ],
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": ["f_x"],
                    "externals": [],
                    "constraints_text": [
                        f"x = {x}",
                        f"threshold: t = {t}",
                    ],
                    "constraints_machine": [
                        {"type": "query_point", "x": x},
                        {"type": "threshold", "value": t},
                    ],
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "threshold",
                    "value": t,
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }
    return instance


def _is_solvable(view: Dict[str, Any]) -> bool:
    cmap = {c["type"] for c in view["private_data"]["constraints_machine"]}
    branch_count = sum(
        1 for c in view["private_data"]["constraints_machine"]
        if c["type"] == "branch_formula"
    )
    return "query_point" in cmap and "threshold" in cmap and branch_count >= 2


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    return not _is_solvable(agent_view)


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "threshold":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "threshold", "value": hint["value"]}
        )
        new_view["private_data"]["constraints_text"].append(
            f"threshold: t = {hint['value']}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    return _is_solvable(agent_view)


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    x = t = None
    branches: Dict[int, Dict[str, int]] = {}
    for v in instance["agent_views"]:
        for c in v["private_data"]["constraints_machine"]:
            if c["type"] == "query_point":
                x = c["x"]
            elif c["type"] == "threshold":
                t = c["value"]
            elif c["type"] == "branch_formula":
                branches[c["branch"]] = {"slope": c["slope"], "intercept": c["intercept"]}
    if x is None or t is None or len(branches) < 2:
        raise ValueError("insufficient data to solve piecewise instance")
    active = 0 if x < t else 1
    b = branches[active]
    return {"value": int(b["slope"] * x + b["intercept"])}