"""Low-rank (rank-1) 2x2 matrix completion with a single missing entry."""
from __future__ import annotations

from typing import Any, Dict, Tuple
import random


FAMILY_NAME = "matrix_completion"
FAMILY_TYPE = "A"  # Fixed hint slot: always position [1][0] in the 2x2 matrix


def _extract_entries(view: Dict[str, Any]) -> Dict[Tuple[int, int], int]:
    entries: Dict[Tuple[int, int], int] = {}
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] == "entry":
            entries[(int(c["row"]), int(c["col"]))] = int(c["value"])
    return entries


def _possible_target(entries: Dict[Tuple[int, int], int]) -> int | None:
    if (0, 0) not in entries or (0, 1) not in entries or (1, 0) not in entries:
        return None
    a = entries[(0, 0)]
    b = entries[(0, 1)]
    c = entries[(1, 0)]
    if a == 0:
        return None
    num = b * c
    if num % a != 0:
        return None
    return num // a


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    value_range = 2 + difficulty

    while True:
        u0 = rng.randint(1, value_range)
        u1 = rng.randint(1, value_range)
        v0 = rng.randint(1, value_range)
        v1 = rng.randint(1, value_range)

        a = u0 * v0
        b = u0 * v1
        c = u1 * v0
        d = u1 * v1
        if a == 0:
            continue

        instance = {
            "id": instance_id,
            "family": FAMILY_NAME,
            "difficulty": difficulty,
            "n_agents": 2,
            "answer_type": "int",
            "global_solution": {"value": int(d)},
            "global_metadata": {
                "shape": [2, 2],
                "target": [1, 1],
            },
            "agent_views": [
                {
                    "agent_id": "A",
                    "private_data": {
                        "unknowns": ["m00", "m01", "m10", "m11"],
                        "externals": [],
                        "constraints_text": [f"m00 = {a}", f"m01 = {b}"],
                        "constraints_machine": [
                            {"type": "entry", "row": 0, "col": 0, "value": a},
                            {"type": "entry", "row": 0, "col": 1, "value": b},
                        ],
                    },
                },
                {
                    "agent_id": "B",
                    "private_data": {
                        "unknowns": ["m00", "m01", "m10", "m11"],
                        "externals": [],
                        "constraints_text": [f"m10 = {c}"],
                        "constraints_machine": [
                            {"type": "entry", "row": 1, "col": 0, "value": c},
                        ],
                    },
                },
            ],
            "minimal_hint_spec": {
                "k_min": 1,
                "atomic_hints": [
                    {
                        "hint_id": "h1",
                        "kind": "entry",
                        "row": 1,
                        "col": 0,
                        "value": c,
                        "providers": ["B"],
                        "consumers": ["A"],
                    }
                ],
            },
        }
        return instance


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    entries = _extract_entries(agent_view)
    return _possible_target(entries) is None


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "entry":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "entry", "row": hint["row"], "col": hint["col"], "value": hint["value"]}
        )
        new_view["private_data"]["constraints_text"].append(
            f"m{hint['row']}{hint['col']} = {hint['value']}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    entries = _extract_entries(agent_view)
    return _possible_target(entries) is not None


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    all_entries: Dict[Tuple[int, int], int] = {}
    for v in instance["agent_views"]:
        all_entries.update(_extract_entries(v))
    target = _possible_target(all_entries)
    if target is None:
        raise ValueError("matrix completion not uniquely solvable")
    return {"value": int(target)}
