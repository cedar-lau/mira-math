"""Deconvolution on short sequences with missing measurement."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random

from mira_math.utils.exact import solve_linear_system


FAMILY_NAME = "deconvolution"
FAMILY_TYPE = "B"  # Variable hint slot: any output measurement index


def _convolve(x: List[int], h: List[int]) -> List[int]:
    n = len(x)
    m = len(h)
    y = [0 for _ in range(n + m - 1)]
    for i in range(n):
        for j in range(m):
            y[i + j] += x[i] * h[j]
    return y


def _format_conv_eq(k: int, h: List[int], rhs: int, n: int) -> str:
    terms = []
    for i, coef in enumerate(h):
        idx = k - i
        if idx < 0 or idx >= n:
            continue
        terms.append(f"{coef}x{idx}")
    lhs = " + ".join(terms).replace("+-", "-")
    return f"{lhs} = {rhs}"


def _build_linear_system(
    constraints: List[Dict[str, Any]],
    h: List[int],
    n: int,
) -> Tuple[List[List[int]], List[int]]:
    A: List[List[int]] = []
    b: List[int] = []
    for c in constraints:
        if c["type"] != "conv_eq":
            continue
        k = int(c["index"])
        row = [0 for _ in range(n)]
        for i, coef in enumerate(h):
            idx = k - i
            if 0 <= idx < n:
                row[idx] += int(coef)
        A.append(row)
        b.append(int(c["value"]))
    return A, b


def _solve_for_x(constraints: List[Dict[str, Any]], h: List[int], n: int) -> Dict[str, Any]:
    A, b = _build_linear_system(constraints, h, n)
    return solve_linear_system(A, b)


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    n = 3 + min(difficulty, 2)
    m = 2 if difficulty <= 1 else 3
    value_range = 2 + difficulty

    while True:
        h = [rng.randint(-value_range, value_range) for _ in range(m)]
        if h[0] == 0:
            continue
        x = [rng.randint(-value_range, value_range) for _ in range(n)]
        y = _convolve(x, h)

        # Find a missing index that makes A ill-posed
        missing_index = None
        for k in range(len(y)):
            constraints = [
                {"type": "conv_eq", "index": i, "value": y[i]}
                for i in range(len(y))
                if i != k
            ]
            res = _solve_for_x(constraints, h, n)
            if res["status"] != "unique":
                missing_index = k
                break
        if missing_index is None:
            continue

        full_constraints = [
            {"type": "conv_eq", "index": i, "value": y[i]} for i in range(len(y))
        ]
        if _solve_for_x(full_constraints, h, n)["status"] != "unique":
            continue

        missing_value = y[missing_index]

        agent_a_constraints = [
            {"type": "conv_eq", "index": i, "value": y[i]}
            for i in range(len(y))
            if i != missing_index
        ]
        agent_a_text = [
            _format_conv_eq(i, h, y[i], n) for i in range(len(y)) if i != missing_index
        ]

        agent_b_constraints = [{"type": "conv_eq", "index": missing_index, "value": missing_value}]
        agent_b_text = [_format_conv_eq(missing_index, h, missing_value, n)]

        instance = {
            "id": instance_id,
            "family": FAMILY_NAME,
            "difficulty": difficulty,
            "n_agents": 2,
            "answer_type": "dict_int",
            "global_solution": {f"x{i}": int(x[i]) for i in range(n)},
            "global_metadata": {
                "kernel": h,
                "length_x": n,
                "length_y": len(y),
                "missing_index": missing_index,
            },
            "agent_views": [
                {
                    "agent_id": "A",
                    "private_data": {
                        "unknowns": [f"x{i}" for i in range(n)],
                        "externals": [],
                        "constraints_text": agent_a_text,
                        "constraints_machine": agent_a_constraints,
                    },
                },
                {
                    "agent_id": "B",
                    "private_data": {
                        "unknowns": [f"x{i}" for i in range(n)],
                        "externals": [],
                        "constraints_text": agent_b_text,
                        "constraints_machine": agent_b_constraints,
                    },
                },
            ],
            "minimal_hint_spec": {
                "k_min": 1,
                "atomic_hints": [
                    {
                        "hint_id": "h1",
                        "kind": "conv_eq",
                        "index": missing_index,
                        "value": missing_value,
                        "providers": ["B"],
                        "consumers": ["A"],
                    }
                ],
            },
        }
        return instance


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    h = instance["global_metadata"]["kernel"]
    n = int(instance["global_metadata"]["length_x"])
    res = _solve_for_x(agent_view["private_data"]["constraints_machine"], h, n)
    return res["status"] != "unique"


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "conv_eq":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "conv_eq", "index": hint["index"], "value": hint["value"]}
        )
        # Text fallback: just state the measurement.
        new_view["private_data"]["constraints_text"].append(
            f"y[{hint['index']}] = {hint['value']}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    h = instance["global_metadata"]["kernel"]
    n = int(instance["global_metadata"]["length_x"])
    res = _solve_for_x(agent_view["private_data"]["constraints_machine"], h, n)
    return res["status"] == "unique"


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    h = instance["global_metadata"]["kernel"]
    n = int(instance["global_metadata"]["length_x"])
    all_constraints = []
    for v in instance["agent_views"]:
        all_constraints.extend(v["private_data"]["constraints_machine"])
    res = _solve_for_x(all_constraints, h, n)
    if res["status"] != "unique":
        raise ValueError("deconvolution not uniquely solvable")
    sol = res["solution"]
    return {f"x{i}": int(sol[i]) for i in range(n)}
