"""Matrix entry recoverable only via full matrix multiplication — not eigenvector.

An n x n integer matrix M has one entry M[missing_row][missing_col] hidden.
A known integer eigenvector v satisfies M * v = lambda * v.

Crucially, v[missing_col] = 0, so the eigenvector equation for row missing_row:
  sum_c M[missing_row][c] * v[c] = lambda * v[missing_row]
does NOT constrain M[missing_row][missing_col] (that term vanishes). Therefore
Agent A cannot recover the missing entry from the eigenvector equation alone.

Agent A holds: all entries of M except M[missing_row][missing_col], plus the
  full eigenvector v (presented as individual component sentences).
Agent B holds: M[missing_row][missing_col].

Query: (M^2)[missing_row][target_col], which equals
  sum_k M[missing_row][k] * M[k][target_col]
and includes the term M[missing_row][missing_col] * M[missing_col][target_col].
Since M[missing_col][target_col] != 0 (enforced in generation), A cannot
compute the query without asking B.

A must: scan all matrix entry sentences, find the gap at (missing_row, missing_col),
verify via the eigenvector equations that this entry is unconstrained, and ask B.

Answer: (M^2)[missing_row][target_col] as an integer.
"""
from __future__ import annotations

from typing import Any, Dict, List
import random

FAMILY_NAME = "eigenvector_missing_entry"
FAMILY_TYPE = "B"  # Variable hint slot: any matrix entry (missing_row, missing_col)

_ENTRY_TEMPLATES = [
    "The entry at row {r}, column {c} of matrix M is {v}.",
    "M[{r}][{c}] = {v}.",
    "Row {r}, column {c} of the coefficient matrix equals {v}.",
    "The ({r},{c}) entry of M is {v}.",
]

_EIGVEC_TEMPLATES = [
    "The {k}-th component of eigenvector v is {val}.",
    "v[{k}] = {val}.",
    "Component {k} of the eigenvector v equals {val}.",
]


def _nl_entry(rng: random.Random, r: int, c: int, v: int) -> str:
    return rng.choice(_ENTRY_TEMPLATES).format(r=r, c=c, v=v)


def _nl_eigvec(rng: random.Random, k: int, val: int) -> str:
    return rng.choice(_EIGVEC_TEMPLATES).format(k=k, val=val)


def _mat_mul(A: List[List[int]], B: List[List[int]], n: int) -> List[List[int]]:
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    n = difficulty + 1   # 2, 3, 4
    max_val = 3 + 2 * (difficulty - 1)  # 3, 5, 7

    while True:
        # missing_col = the zero index of v
        missing_col = rng.randint(0, n - 1)
        missing_row = rng.randint(0, n - 1)

        # Eigenvector v: v[missing_col] = 0, all others = 1..3
        v = [rng.randint(1, 3) if k != missing_col else 0 for k in range(n)]

        # Eigenvalue lambda
        lam = rng.randint(1, max_val)

        # Build M: for each row k, choose M[k][c] for c != missing_col as integers
        # satisfying sum_{c != missing_col} M[k][c] * v[c] = lam * v[k].
        # M[k][missing_col] is free (unconstrained by eigenvector).
        non_j_cols = [c for c in range(n) if c != missing_col]

        M = [[0] * n for _ in range(n)]
        ok = True
        for k in range(n):
            if len(non_j_cols) == 1:
                # Only one column outside missing_col; must satisfy:
                # M[k][non_j_cols[0]] * v[non_j_cols[0]] = lam * v[k]
                target = lam * v[k]
                divisor = v[non_j_cols[0]]
                if divisor == 0 or target % divisor != 0:
                    ok = False
                    break
                M[k][non_j_cols[0]] = target // divisor
            else:
                # Pick free values for all but the last non_j col, solve for last
                free_cols = non_j_cols[:-1]
                last_col = non_j_cols[-1]
                free_vals = [rng.randint(-max_val, max_val) for _ in free_cols]
                for idx, c in enumerate(free_cols):
                    M[k][c] = free_vals[idx]
                rhs = lam * v[k] - sum(M[k][c] * v[c] for c in free_cols)
                denom = v[last_col]
                if denom == 0 or rhs % denom != 0:
                    ok = False
                    break
                M[k][last_col] = rhs // denom

            # Free entry in missing_col column
            M[k][missing_col] = rng.randint(-max_val, max_val)

        if not ok:
            continue

        # Verify Mv = lam*v
        for k in range(n):
            dot = sum(M[k][c] * v[c] for c in range(n))
            if dot != lam * v[k]:
                ok = False
                break
        if not ok:
            continue

        # Query: (M^2)[missing_row][target_col]
        target_col = rng.randint(0, n - 1)

        # Check that M[missing_col][target_col] != 0 (so A actually needs the missing entry)
        if M[missing_col][target_col] == 0:
            continue

        # Compute M^2
        M2 = _mat_mul(M, M, n)
        answer = M2[missing_row][target_col]

        # Keep answers in a reasonable range
        if abs(answer) > 10000:
            continue

        break

    # Build A's constraints
    a_text: List[str] = []
    a_machine: List[Dict[str, Any]] = []

    for r in range(n):
        for c in range(n):
            if r == missing_row and c == missing_col:
                continue
            a_text.append(_nl_entry(rng, r, c, M[r][c]))
            a_machine.append({"type": "matrix_entry", "row": r, "col": c, "value": M[r][c]})

    for k in range(n):
        a_text.append(_nl_eigvec(rng, k, v[k]))
        a_machine.append({"type": "eigvec", "index": k, "value": v[k]})

    # B: just the missing entry
    hint_val = M[missing_row][missing_col]
    b_text = [f"M[{missing_row}][{missing_col}] = {hint_val}"]
    b_machine = [{"type": "matrix_entry", "row": missing_row, "col": missing_col, "value": hint_val}]

    return {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "int",
        "global_solution": {"value": answer},
        "global_metadata": {
            "n": n,
            "missing_row": missing_row,
            "missing_col": missing_col,
            "target_col": target_col,
            "eigenvalue": lam,
            "M": M,
            "v": v,
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": [f"(M^2)[{missing_row}][{target_col}]"],
                    "externals": [],
                    "constraints_text": a_text,
                    "constraints_machine": a_machine,
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": [f"(M^2)[{missing_row}][{target_col}]"],
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
                    "kind": "matrix_entry",
                    "row": missing_row,
                    "col": missing_col,
                    "value": hint_val,
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }


def _known_entries(view: Dict[str, Any]) -> set:
    return {
        (c["row"], c["col"])
        for c in view["private_data"]["constraints_machine"]
        if c["type"] == "matrix_entry"
    }


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n"]
    known = _known_entries(agent_view)
    return not all((r, c) in known for r in range(n) for c in range(n))


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "matrix_entry":
        r, c, v = hint["row"], hint["col"], hint["value"]
        new_view["private_data"]["constraints_machine"].append(
            {"type": "matrix_entry", "row": r, "col": c, "value": v}
        )
        new_view["private_data"]["constraints_text"].append(f"M[{r}][{c}] = {v}")
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n"]
    known = _known_entries(agent_view)
    return all((r, c) in known for r in range(n) for c in range(n))


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    n = instance["global_metadata"]["n"]
    M = instance["global_metadata"]["M"]
    mr = instance["global_metadata"]["missing_row"]
    tc = instance["global_metadata"]["target_col"]
    M2 = _mat_mul(M, M, n)
    return {"value": M2[mr][tc]}
