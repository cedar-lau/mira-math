"""Markov chain with one missing transition probability.

n-state Markov chain (n = difficulty + 1, capped at 4).

Split:
  Agent A holds:
    - every row of the transition matrix EXCEPT the missing row, completely
    - (n-2) specific entries of the missing row  (enough so that ONE more
      entry from B lets A derive the last entry via the row-sum=1 constraint)
  Agent B holds:
    - exactly ONE specific entry of the missing row  (hint_col)

After receiving the hint, A knows (n-1) entries of the missing row and
derives the nth via row-sum=1. The full matrix determines a unique
steady-state distribution; A reports pi(target_state) as a reduced fraction.

Which row is missing and which column B holds are both randomised per
instance, so the missing probability is never always the same.

Answer: pi(target_state) as a reduced fraction {numerator, denominator}.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List
import random


FAMILY_NAME = "markov_missing_transition"
FAMILY_TYPE = "B"  # Variable hint slot: any transition entry (missing_row, hint_col)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frac_text(f: Fraction) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


_NL_TEMPLATES = [
    "From state {i}, there is a {p} chance of transitioning to state {j}.",
    "The probability of moving from state {i} to state {j} is {p}.",
    "A chain in state {i} moves to state {j} with probability {p}.",
    "State {i} transitions to state {j} with probability {p}.",
    "Starting in state {i}, the chain goes to state {j} with probability {p}.",
]


def _nl_transition(rng: random.Random, i: int, j: int, f: Fraction) -> str:
    return rng.choice(_NL_TEMPLATES).format(i=i, j=j, p=_frac_text(f))


def _random_row(rng: random.Random, n: int, max_part: int) -> List[Fraction]:
    """Return a row of n positive fractions that sum to 1."""
    while True:
        parts = [rng.randint(1, max_part) for _ in range(n)]
        total = sum(parts)
        row = [Fraction(p, total) for p in parts]
        if all(f > 0 for f in row):
            return row


def _steady_state(P: List[List[Fraction]], n: int) -> List[Fraction]:
    """Solve pi*P = pi with sum(pi) = 1 via Gauss-Jordan over Fraction."""
    # Build augmented matrix: (n-1) balance rows + 1 normalisation row
    mat: List[List[Fraction]] = []
    for i in range(n - 1):
        row = [P[j][i] - (Fraction(1) if j == i else Fraction(0))
               for j in range(n)]
        row.append(Fraction(0))
        mat.append(row)
    mat.append([Fraction(1)] * n + [Fraction(1)])

    # Gauss-Jordan
    for col in range(n):
        pivot = next((r for r in range(col, n) if mat[r][col] != 0), None)
        if pivot is None:
            raise ValueError("Singular transition matrix")
        mat[col], mat[pivot] = mat[pivot], mat[col]
        p = mat[col][col]
        mat[col] = [x / p for x in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [mat[r][k] - f * mat[col][k] for k in range(n + 1)]

    return [mat[i][n] for i in range(n)]


# ---------------------------------------------------------------------------
# Family interface
# ---------------------------------------------------------------------------

def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    n = min(difficulty + 1, 4)       # 2, 3, 4 for difficulty 1, 2, 3
    max_part = 2 + difficulty        # controls fraction complexity

    while True:
        # --- generate full transition matrix ---
        P = [_random_row(rng, n, max_part) for _ in range(n)]

        # --- randomise the split ---
        missing_row = rng.randint(0, n - 1)

        # Choose which columns go to A (n-2), B (1), and are derived (1).
        all_cols = list(range(n))
        rng.shuffle(all_cols)
        a_cols = sorted(all_cols[:n - 2])   # A knows these from missing_row
        hint_col = all_cols[n - 2]          # B holds this entry
        derived_col = all_cols[n - 1]       # A derives this from row-sum

        # --- pick target state ---
        target_state = rng.randint(0, n - 1)

        # --- compute ground-truth steady state ---
        try:
            pi = _steady_state(P, n)
        except ValueError:
            continue

        answer = pi[target_state]
        if answer <= 0 or answer >= 1:
            continue
        # Keep denominator manageable for LLM arithmetic
        if answer.denominator > 200:
            continue

        break

    # --- build constraints for each agent ---
    # A: all complete rows + a_cols entries of missing_row
    a_machine: List[Dict[str, Any]] = []
    a_text: List[str] = []
    for i in range(n):
        if i == missing_row:
            for j in a_cols:
                f = P[i][j]
                a_machine.append({"type": "transition", "from_state": i, "to_state": j,
                                   "num": int(f.numerator), "denom": int(f.denominator)})
                a_text.append(_nl_transition(rng, i, j, f))
        else:
            for j in range(n):
                f = P[i][j]
                a_machine.append({"type": "transition", "from_state": i, "to_state": j,
                                   "num": int(f.numerator), "denom": int(f.denominator)})
                a_text.append(_nl_transition(rng, i, j, f))

    # B: only the hint entry
    f_hint = P[missing_row][hint_col]
    b_machine = [{"type": "transition", "from_state": missing_row, "to_state": hint_col,
                  "num": int(f_hint.numerator), "denom": int(f_hint.denominator)}]
    b_text = [f"P({missing_row} -> {hint_col}) = {_frac_text(f_hint)}"]

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "dict_int",
        "global_solution": {
            "numerator": int(answer.numerator),
            "denominator": int(answer.denominator),
        },
        "global_metadata": {
            "n_states": n,
            "missing_row": missing_row,
            "hint_col": hint_col,
            "derived_col": derived_col,
            "a_cols": a_cols,
            "target_state": target_state,
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": [f"pi_{target_state}"],
                    "externals": [],
                    "constraints_text": a_text,
                    "constraints_machine": a_machine,
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": [f"pi_{target_state}"],
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
                    "kind": "transition",
                    "from_state": missing_row,
                    "to_state": hint_col,
                    "num": int(f_hint.numerator),
                    "denom": int(f_hint.denominator),
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }
    return instance


def _row_entries(view: Dict[str, Any], row: int) -> Dict[int, Fraction]:
    """Return {col: Fraction} for all known transitions from a given row."""
    return {
        c["to_state"]: Fraction(c["num"], c["denom"])
        for c in view["private_data"]["constraints_machine"]
        if c["type"] == "transition" and c["from_state"] == row
    }


def _can_solve(view: Dict[str, Any], n: int) -> bool:
    """True when every row has at least (n-1) known entries (last derivable via sum=1)."""
    for i in range(n):
        if len(_row_entries(view, i)) < n - 1:
            return False
    return True


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_states"]
    return not _can_solve(agent_view, n)


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "transition":
        fs, ts = hint["from_state"], hint["to_state"]
        f = Fraction(hint["num"], hint["denom"])
        new_view["private_data"]["constraints_machine"].append(
            {"type": "transition", "from_state": fs, "to_state": ts,
             "num": hint["num"], "denom": hint["denom"]}
        )
        new_view["private_data"]["constraints_text"].append(
            f"P({fs} -> {ts}) = {_frac_text(f)}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_states"]
    return _can_solve(agent_view, n)


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    n = instance["global_metadata"]["n_states"]
    target = instance["global_metadata"]["target_state"]

    # Collect all known transition entries across both views
    known: Dict[tuple, Fraction] = {}
    for v in instance["agent_views"]:
        for c in v["private_data"]["constraints_machine"]:
            if c["type"] == "transition":
                known[(c["from_state"], c["to_state"])] = Fraction(c["num"], c["denom"])

    # Build full n×n matrix, deriving the one missing entry per row from sum=1
    P: List[List[Fraction]] = []
    for i in range(n):
        row_known = {j: known[(i, j)] for j in range(n) if (i, j) in known}
        missing = [j for j in range(n) if j not in row_known]
        if len(missing) == 1:
            row_known[missing[0]] = Fraction(1) - sum(row_known.values())
        elif len(missing) > 1:
            raise ValueError(f"Row {i} has {len(missing)} unknown entries")
        P.append([row_known[j] for j in range(n)])

    pi = _steady_state(P, n)
    result = pi[target]
    return {"numerator": int(result.numerator), "denominator": int(result.denominator)}