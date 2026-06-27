"""CRT reconstruction with missing congruence."""
from __future__ import annotations

from typing import Any, Dict, List
import random

from mira_math.utils.crt import crt_solve, product_moduli


FAMILY_NAME = "crt_reconstruction"
FAMILY_TYPE = "A"  # Fixed hint slot: always the 3rd congruence


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    primes = [3, 5, 7, 11, 13, 17, 19]
    rng.shuffle(primes)
    moduli = primes[:3]
    M = 1
    for m in moduli:
        M *= m

    x = rng.randint(0, M - 1)
    congruences = [(m, x % m) for m in moduli]

    agent_a = congruences[:2]
    agent_b = congruences[2:]

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "dict_int",
        "global_solution": {"x": x},
        "global_metadata": {
            "variable": "x",
            "range": [0, M],
            "moduli": moduli,
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": ["x"],
                    "externals": [],
                    "constraints_text": [f"x ≡ {r} (mod {m})" for m, r in agent_a],
                    "constraints_machine": [
                        {"type": "congruence", "var": "x", "mod": m, "residue": r}
                        for m, r in agent_a
                    ],
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": ["x"],
                    "externals": [],
                    "constraints_text": [f"x ≡ {r} (mod {m})" for m, r in agent_b],
                    "constraints_machine": [
                        {"type": "congruence", "var": "x", "mod": m, "residue": r}
                        for m, r in agent_b
                    ],
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "congruence",
                    "var": "x",
                    "mod": agent_b[0][0],
                    "residue": agent_b[0][1],
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }
    return instance


def _extract_congruences(view: Dict[str, Any]) -> List[tuple[int, int]]:
    congruences = []
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] == "congruence":
            congruences.append((int(c["mod"]), int(c["residue"])))
    return congruences


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    congruences = _extract_congruences(agent_view)
    if not congruences:
        return True
    M_total = 1
    for m in instance["global_metadata"]["moduli"]:
        M_total *= m
    M_view = product_moduli(congruences)
    return M_total // M_view > 1


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "congruence":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "congruence", "var": hint["var"], "mod": hint["mod"], "residue": hint["residue"]}
        )
        new_view["private_data"]["constraints_text"].append(
            f"{hint['var']} ≡ {hint['residue']} (mod {hint['mod']})"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    congruences = _extract_congruences(agent_view)
    if not congruences:
        return False
    M_total = 1
    for m in instance["global_metadata"]["moduli"]:
        M_total *= m
    M_view = product_moduli(congruences)
    return M_total // M_view == 1


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    all_congruences: List[tuple[int, int]] = []
    for v in instance["agent_views"]:
        all_congruences.extend(_extract_congruences(v))
    x, M = crt_solve(all_congruences)
    if not (0 <= x < M):
        raise ValueError("CRT solution out of range")
    return {"x": int(x)}

