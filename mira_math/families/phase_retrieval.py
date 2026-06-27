"""Toy phase retrieval with DFT magnitudes and one sign hint."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random


FAMILY_NAME = "phase_retrieval"
FAMILY_TYPE = "A"  # Fixed hint slot: always the sign of x[0]


def _fft_mag_sq(x: List[int]) -> List[int]:
    x0, x1, x2, x3 = x
    m0 = (x0 + x1 + x2 + x3) ** 2
    m1 = (x0 - x2) ** 2 + (x3 - x1) ** 2
    m2 = (x0 - x1 + x2 - x3) ** 2
    m3 = (x0 - x2) ** 2 + (x1 - x3) ** 2
    return [m0, m1, m2, m3]


def _candidates(mags: List[int] | None, sign: int | None, value_range: int) -> List[List[int]]:
    if mags is None:
        return []
    candidates: List[List[int]] = []
    for x0 in range(-value_range, value_range + 1):
        for x1 in range(-value_range, value_range + 1):
            for x2 in range(-value_range, value_range + 1):
                for x3 in range(-value_range, value_range + 1):
                    x = [x0, x1, x2, x3]
                    if _fft_mag_sq(x) != mags:
                        continue
                    if sign is not None:
                        if x0 == 0 or (1 if x0 > 0 else -1) != sign:
                            continue
                    candidates.append(x)
    return candidates


def _extract_constraints(view: Dict[str, Any]) -> Tuple[List[int] | None, int | None]:
    mags = None
    sign = None
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] == "fft_mag_sq":
            mags = [int(v) for v in c["values"]]
        elif c["type"] == "sign":
            sign = int(c["sign"])
    return mags, sign


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    value_range = 2 + min(difficulty, 2)

    while True:
        x = [rng.randint(-value_range, value_range) for _ in range(4)]
        if x[0] == 0:
            continue
        mags = _fft_mag_sq(x)
        candidates = _candidates(mags, None, value_range)
        if len(candidates) <= 1:
            continue
        sign = 1 if x[0] > 0 else -1
        candidates_with_sign = _candidates(mags, sign, value_range)
        if len(candidates_with_sign) != 1:
            continue
        break

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "dict_int",
        "global_solution": {f"x{i}": int(x[i]) for i in range(4)},
        "global_metadata": {
            "N": 4,
            "value_range": value_range,
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": ["x0", "x1", "x2", "x3"],
                    "externals": [],
                    "constraints_text": [f"|FFT(x)|^2 = {mags}"],
                    "constraints_machine": [
                        {"type": "fft_mag_sq", "values": mags}
                    ],
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": ["x0", "x1", "x2", "x3"],
                    "externals": [],
                    "constraints_text": [f"sign(x0) = {sign}"],
                    "constraints_machine": [
                        {"type": "sign", "index": 0, "sign": sign}
                    ],
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "sign",
                    "index": 0,
                    "sign": sign,
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }
    return instance


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    value_range = int(instance["global_metadata"]["value_range"])
    mags, sign = _extract_constraints(agent_view)
    if mags is None:
        return True
    candidates = _candidates(mags, sign, value_range)
    return len(candidates) != 1


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "sign":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "sign", "index": hint["index"], "sign": hint["sign"]}
        )
        new_view["private_data"]["constraints_text"].append(
            f"sign(x{hint['index']}) = {hint['sign']}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    value_range = int(instance["global_metadata"]["value_range"])
    mags, sign = _extract_constraints(agent_view)
    if mags is None:
        return False
    candidates = _candidates(mags, sign, value_range)
    return len(candidates) == 1


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    value_range = int(instance["global_metadata"]["value_range"])
    mags = None
    sign = None
    for v in instance["agent_views"]:
        m, s = _extract_constraints(v)
        if m is not None:
            mags = m
        if s is not None:
            sign = s
    candidates = _candidates(mags, sign, value_range)
    if len(candidates) != 1:
        raise ValueError("phase retrieval not uniquely solvable")
    x = candidates[0]
    return {f"x{i}": int(x[i]) for i in range(4)}
