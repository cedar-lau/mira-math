"""Series/parallel resistor network with one resistor value unknown.

A binary tree of series and parallel combinations. Leaves are named resistors
R1, R2, ... Each resistor has a positive integer resistance (in ohms).

Agent A holds: the full circuit topology as natural-language sentences, plus
  the resistance value of every resistor EXCEPT one leaf R_k (randomised).
Agent B holds: R_k (the one missing resistance).

A must scan all resistor value sentences, find the missing R_k, then ask B.
After receiving R_k, A evaluates the series/parallel tree to compute R_eq and
reports it as a reduced fraction.

Why non-trivial: The missing leaf can be at any depth in the tree. A must trace
the circuit structure (series/parallel reduction) to locate the undetermined
sub-expression, then identify which leaf feeds it — requiring understanding of
series/parallel formulas.

Answer: R_eq as a fully reduced fraction {numerator, denominator}.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Tuple
import random

FAMILY_NAME = "circuit_missing_resistance"
FAMILY_TYPE = "B"  # Variable hint slot: any one of n resistors R_k

# Tree node: ("leaf", idx) or ("series"/"parallel", left, right)
Tree = Any


def _build_tree(rng: random.Random, leaf_ids: List[int]) -> Tree:
    """Recursively build a random binary series/parallel tree."""
    if len(leaf_ids) == 1:
        return ("leaf", leaf_ids[0])
    # Split into two groups (left-biased for smaller trees)
    k = rng.randint(1, len(leaf_ids) - 1)
    left = _build_tree(rng, leaf_ids[:k])
    right = _build_tree(rng, leaf_ids[k:])
    op = rng.choice(["series", "parallel"])
    return (op, left, right)


def _compute_resistance(tree: Tree, values: Dict[int, Fraction]) -> Fraction:
    if tree[0] == "leaf":
        return values[tree[1]]
    op, left, right = tree
    R_l = _compute_resistance(left, values)
    R_r = _compute_resistance(right, values)
    if op == "series":
        return R_l + R_r
    else:
        return R_l * R_r / (R_l + R_r)


def _describe_tree(tree: Tree, groups: List[str], resistors: Dict[int, str]) -> Tuple[str, List[str]]:
    """Return (component_name, list_of_description_sentences) for a subtree."""
    if tree[0] == "leaf":
        return resistors[tree[1]], []
    op, left, right = tree
    left_name, left_lines = _describe_tree(left, groups, resistors)
    right_name, right_lines = _describe_tree(right, groups, resistors)
    is_last = len(groups) == 0 and left[0] != "leaf" and right[0] != "leaf"
    group_id = len(groups) + 1
    group_name = f"G{group_id}"
    groups.append(group_name)
    op_word = "series" if op == "series" else "parallel"
    line = f"{left_name} and {right_name} are connected in {op_word}, forming group {group_name}."
    return group_name, left_lines + right_lines + [line]


def _describe_tree_top(tree: Tree) -> List[str]:
    """Describe the circuit topology from bottom up, returning all sentences."""
    groups: List[str] = []
    resistor_names = {}

    def collect_leaves(t):
        if t[0] == "leaf":
            resistor_names[t[1]] = f"R{t[1] + 1}"
        else:
            collect_leaves(t[1])
            collect_leaves(t[2])

    collect_leaves(tree)

    name, lines = _describe_tree(tree, groups, resistor_names)
    if not lines:
        # Single resistor — degenerate case
        return [f"The circuit consists of a single resistor {name}."]
    lines[-1] = lines[-1].replace(f"forming group {groups[-1]}.", "forming the total equivalent resistance.")
    return lines


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    n_resistors = 2 * difficulty + 1   # 3, 5, 7 for difficulties 1, 2, 3
    max_val = 5 + 3 * (difficulty - 1)  # 5, 8, 11

    while True:
        leaf_ids = list(range(n_resistors))
        rng.shuffle(leaf_ids)
        tree = _build_tree(rng, leaf_ids)

        # Assign random positive integer resistances
        values_int = {i: rng.randint(1, max_val) for i in range(n_resistors)}
        values = {i: Fraction(v) for i, v in values_int.items()}

        # Pick missing resistor
        missing_idx = rng.randint(0, n_resistors - 1)

        try:
            R_eq = _compute_resistance(tree, values)
        except ZeroDivisionError:
            continue

        if R_eq.denominator > 500:
            continue
        if R_eq <= 0:
            continue

        break

    # Build topology description
    topo_lines = _describe_tree_top(tree)

    # Build A's constraints: topology + all resistance values except missing_idx
    a_text: List[str] = list(topo_lines)
    a_machine: List[Dict[str, Any]] = []

    # Encode topology in machine format (simplified: just record the tree structure)
    a_machine.append({"type": "topology", "description": topo_lines})

    for i in range(n_resistors):
        if i == missing_idx:
            continue
        v = values_int[i]
        a_text.append(f"Resistor R{i + 1} has resistance {v} ohms.")
        a_machine.append({"type": "resistance", "resistor": i, "value": v})

    # B: just the missing resistance
    hint_val = values_int[missing_idx]
    b_text = [f"R{missing_idx + 1} = {hint_val} ohms"]
    b_machine = [{"type": "resistance", "resistor": missing_idx, "value": hint_val}]

    # Encode tree as a serialisable structure for solve_global
    def encode_tree(t):
        if t[0] == "leaf":
            return ["leaf", t[1]]
        return [t[0], encode_tree(t[1]), encode_tree(t[2])]

    return {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "rational",
        "global_solution": {
            "numerator": int(R_eq.numerator),
            "denominator": int(R_eq.denominator),
        },
        "global_metadata": {
            "n_resistors": n_resistors,
            "missing_resistor": missing_idx,
            "values": values_int,
            "tree": encode_tree(tree),
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": ["R_eq"],
                    "externals": [],
                    "constraints_text": a_text,
                    "constraints_machine": a_machine,
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": ["R_eq"],
                    "externals": [],
                    "constraints_text": b_text,
                    "constraints_machine": b_machine,
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "resistance",
                    "resistor": missing_idx,
                    "value": hint_val,
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }


def _known_resistors(view: Dict[str, Any]) -> set:
    return {
        c["resistor"]
        for c in view["private_data"]["constraints_machine"]
        if c["type"] == "resistance"
    }


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_resistors"]
    known = _known_resistors(agent_view)
    return not all(i in known for i in range(n))


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "resistance":
        idx, v = hint["resistor"], hint["value"]
        new_view["private_data"]["constraints_machine"].append(
            {"type": "resistance", "resistor": idx, "value": v}
        )
        new_view["private_data"]["constraints_text"].append(
            f"Resistor R{idx + 1} has resistance {v} ohms."
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_resistors"]
    known = _known_resistors(agent_view)
    return all(i in known for i in range(n))


def _decode_tree(t) -> Tree:
    if t[0] == "leaf":
        return ("leaf", t[1])
    return (t[0], _decode_tree(t[1]), _decode_tree(t[2]))


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    values = {int(k): Fraction(v) for k, v in instance["global_metadata"]["values"].items()}
    tree = _decode_tree(instance["global_metadata"]["tree"])
    R_eq = _compute_resistance(tree, values)
    return {"numerator": int(R_eq.numerator), "denominator": int(R_eq.denominator)}
