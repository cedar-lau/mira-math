"""Second-order recurrence with missing initial condition."""
from __future__ import annotations

from typing import Any, Dict, List
import random


FAMILY_NAME = "recurrence_missing_init"
FAMILY_TYPE = "A"  # Fixed hint slot: always a(1)


def _compute_sequence(r1: int, r2: int, a0: int, a1: int, n: int) -> int:
    if n == 0:
        return a0
    if n == 1:
        return a1
    prev2, prev1 = a0, a1
    for _ in range(2, n + 1):
        nxt = r1 * prev1 + r2 * prev2
        prev2, prev1 = prev1, nxt
    return prev1


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    r1 = rng.randint(1, 3 + difficulty)
    r2 = rng.randint(-2, 2)
    a0 = rng.randint(-3 - difficulty, 3 + difficulty)
    a1 = rng.randint(-3 - difficulty, 3 + difficulty)
    target_n = rng.randint(4, 7 + difficulty)
    target_value = _compute_sequence(r1, r2, a0, a1, target_n)

    recurrence_constraint = {"type": "recurrence", "coeffs": [r1, r2]}

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "int",
        "global_solution": {"value": int(target_value)},
        "global_metadata": {
            "r1": r1,
            "r2": r2,
            "target_n": target_n,
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": ["a_n"],
                    "externals": [],
                    "constraints_text": [
                        f"a(n+2) = {r1} a(n+1) + {r2} a(n)",
                        f"a(0) = {a0}",
                    ],
                    "constraints_machine": [
                        recurrence_constraint,
                        {"type": "init", "index": 0, "value": a0},
                    ],
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": ["a_n"],
                    "externals": [],
                    "constraints_text": [
                        f"a(n+2) = {r1} a(n+1) + {r2} a(n)",
                        f"a(1) = {a1}",
                    ],
                    "constraints_machine": [
                        recurrence_constraint,
                        {"type": "init", "index": 1, "value": a1},
                    ],
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "init",
                    "index": 1,
                    "value": a1,
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }
    return instance


def _extract_inits(view: Dict[str, Any]) -> Dict[int, int]:
    inits: Dict[int, int] = {}
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] == "init":
            inits[int(c["index"])] = int(c["value"])
    return inits


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    inits = _extract_inits(agent_view)
    return not (0 in inits and 1 in inits)


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "init":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "init", "index": hint["index"], "value": hint["value"]}
        )
        new_view["private_data"]["constraints_text"].append(
            f"a({hint['index']}) = {hint['value']}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    inits = _extract_inits(agent_view)
    return 0 in inits and 1 in inits


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    r1 = int(instance["global_metadata"]["r1"])
    r2 = int(instance["global_metadata"]["r2"])
    target_n = int(instance["global_metadata"]["target_n"])
    inits: Dict[int, int] = {}
    for v in instance["agent_views"]:
        inits.update(_extract_inits(v))
    if 0 not in inits or 1 not in inits:
        raise ValueError("missing initial conditions")
    value = _compute_sequence(r1, r2, inits[0], inits[1], target_n)
    return {"value": int(value)}

