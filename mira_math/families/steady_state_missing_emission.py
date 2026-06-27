"""Hidden Markov Model: steady-state observation probability with one emission missing.

An HMM has n states and m symbols.
- Transition matrix P (n x n) is fully known to Agent A.
- Emission matrix e[state][symbol] (n x m, rows sum to 1) is partially known:
  exactly one entry e[missing_state][target_symbol] is absent from A's view.
- Agent B holds that one emission probability.

A must:
  1. Compute the steady-state distribution pi from P (fully determined).
  2. Try to evaluate P(observe target_symbol) = sum_s pi[s] * e[s][target_symbol].
  3. Discover that e[missing_state][target_symbol] is absent and ask B for it.
After hint: complete the emission-weighted sum and report.

Why non-trivial: A must first solve the balance equations (as in Markov), then
discover the gap only when evaluating the emission sum. The computation precedes
the identification of the missing piece.

Answer: P(observe target_symbol) as a fully reduced fraction {numerator, denominator}.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List
import random

FAMILY_NAME = "steady_state_missing_emission"
FAMILY_TYPE = "B"  # Variable hint slot: any emission probability e[state][symbol]

_TRANS_TEMPLATES = [
    "From state {i}, there is a {p} chance of transitioning to state {j}.",
    "The probability of moving from state {i} to state {j} is {p}.",
    "A chain in state {i} moves to state {j} with probability {p}.",
    "State {i} transitions to state {j} with probability {p}.",
]

_EMIT_TEMPLATES = [
    "The probability of emitting symbol {o} from state {s} is {e}.",
    "State {s} emits symbol {o} with probability {e}.",
    "From state {s}, symbol {o} is observed with probability {e}.",
    "e({s}, {o}) = {e}.",
]


def _frac_text(f: Fraction) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _nl_trans(rng: random.Random, i: int, j: int, p: Fraction) -> str:
    return rng.choice(_TRANS_TEMPLATES).format(i=i, j=j, p=_frac_text(p))


def _nl_emit(rng: random.Random, s: int, o: int, e: Fraction) -> str:
    return rng.choice(_EMIT_TEMPLATES).format(s=s, o=o, e=_frac_text(e))


def _random_stochastic_row(rng: random.Random, n: int, max_part: int) -> List[Fraction]:
    while True:
        parts = [rng.randint(1, max_part) for _ in range(n)]
        total = sum(parts)
        row = [Fraction(p, total) for p in parts]
        if all(f > 0 for f in row):
            return row


def _steady_state(P: List[List[Fraction]], n: int) -> List[Fraction]:
    """Solve pi * P = pi with sum(pi) = 1 via Gauss-Jordan over Fraction."""
    mat: List[List[Fraction]] = []
    for i in range(n - 1):
        row = [P[j][i] - (Fraction(1) if j == i else Fraction(0)) for j in range(n)]
        row.append(Fraction(0))
        mat.append(row)
    mat.append([Fraction(1)] * n + [Fraction(1)])

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


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    n = difficulty + 1    # 2, 3, 4 states
    m = min(difficulty + 1, 3)  # 2, 3, 3 symbols
    max_part = 2 + difficulty

    while True:
        # Transition matrix
        P = [_random_stochastic_row(rng, n, max_part) for _ in range(n)]

        try:
            pi = _steady_state(P, n)
        except ValueError:
            continue

        if any(p <= 0 for p in pi):
            continue

        # Emission matrix: each row is a distribution over m symbols
        emit_parts = [[rng.randint(1, 4) for _ in range(m)] for _ in range(n)]
        emit_totals = [sum(row) for row in emit_parts]
        E = [[Fraction(emit_parts[s][o], emit_totals[s]) for o in range(m)] for s in range(n)]

        # Pick missing: which state's emission for the target symbol is hidden
        target_symbol = rng.randint(0, m - 1)
        missing_state = rng.randint(0, n - 1)

        # Compute observation probability P(observe target_symbol)
        obs_prob = sum(pi[s] * E[s][target_symbol] for s in range(n))

        if obs_prob.denominator > 500:
            continue
        if obs_prob <= 0:
            continue

        break

    # Build A's constraints
    a_text: List[str] = []
    a_machine: List[Dict[str, Any]] = []

    # All transitions (fully known)
    for i in range(n):
        for j in range(n):
            p = P[i][j]
            a_text.append(_nl_trans(rng, i, j, p))
            a_machine.append({"type": "transition", "from_state": i, "to_state": j,
                               "num": int(p.numerator), "denom": int(p.denominator)})

    # Emissions: all except E[missing_state][target_symbol]
    for s in range(n):
        for o in range(m):
            if s == missing_state and o == target_symbol:
                continue
            e_val = E[s][o]
            a_text.append(_nl_emit(rng, s, o, e_val))
            a_machine.append({"type": "emission", "state": s, "symbol": o,
                               "num": int(e_val.numerator), "denom": int(e_val.denominator)})

    # B: just the missing emission
    e_hint = E[missing_state][target_symbol]
    b_text = [f"e({missing_state}, {target_symbol}) = {_frac_text(e_hint)}"]
    b_machine = [{"type": "emission", "state": missing_state, "symbol": target_symbol,
                  "num": int(e_hint.numerator), "denom": int(e_hint.denominator)}]

    # Store P as list of [num, denom] pairs and E as list of [num, denom] pairs
    P_stored = [[[int(P[i][j].numerator), int(P[i][j].denominator)] for j in range(n)] for i in range(n)]
    E_stored = [[[int(E[s][o].numerator), int(E[s][o].denominator)] for o in range(m)] for s in range(n)]

    return {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "rational",
        "global_solution": {
            "numerator": int(obs_prob.numerator),
            "denominator": int(obs_prob.denominator),
        },
        "global_metadata": {
            "n_states": n,
            "n_symbols": m,
            "missing_state": missing_state,
            "target_symbol": target_symbol,
            "P": P_stored,
            "E": E_stored,
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": [f"P(observe symbol {target_symbol})"],
                    "externals": [],
                    "constraints_text": a_text,
                    "constraints_machine": a_machine,
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": [f"P(observe symbol {target_symbol})"],
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
                    "kind": "emission",
                    "state": missing_state,
                    "symbol": target_symbol,
                    "num": int(e_hint.numerator),
                    "denom": int(e_hint.denominator),
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }


def _known_emissions(view: Dict[str, Any]) -> set:
    return {
        (c["state"], c["symbol"])
        for c in view["private_data"]["constraints_machine"]
        if c["type"] == "emission"
    }


def _has_all_data(view: Dict[str, Any], n: int, m: int, ms: int, ts: int) -> bool:
    known_emits = _known_emissions(view)
    if (ms, ts) not in known_emits:
        return False
    machine = view["private_data"]["constraints_machine"]
    known_trans = {(c["from_state"], c["to_state"])
                   for c in machine if c["type"] == "transition"}
    return all((i, j) in known_trans for i in range(n) for j in range(n))


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_states"]
    m = instance["global_metadata"]["n_symbols"]
    ms = instance["global_metadata"]["missing_state"]
    ts = instance["global_metadata"]["target_symbol"]
    return not _has_all_data(agent_view, n, m, ms, ts)


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "emission":
        s, o = hint["state"], hint["symbol"]
        e_val = Fraction(hint["num"], hint["denom"])
        new_view["private_data"]["constraints_machine"].append(
            {"type": "emission", "state": s, "symbol": o,
             "num": hint["num"], "denom": hint["denom"]}
        )
        new_view["private_data"]["constraints_text"].append(
            f"e({s}, {o}) = {_frac_text(e_val)}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_states"]
    m = instance["global_metadata"]["n_symbols"]
    ms = instance["global_metadata"]["missing_state"]
    ts = instance["global_metadata"]["target_symbol"]
    return _has_all_data(agent_view, n, m, ms, ts)


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    n = instance["global_metadata"]["n_states"]
    ts = instance["global_metadata"]["target_symbol"]
    P_stored = instance["global_metadata"]["P"]
    E_stored = instance["global_metadata"]["E"]
    P = [[Fraction(*P_stored[i][j]) for j in range(n)] for i in range(n)]
    E = [[Fraction(*E_stored[s][o]) for o in range(instance["global_metadata"]["n_symbols"])]
         for s in range(n)]
    pi = _steady_state(P, n)
    obs_prob = sum(pi[s] * E[s][ts] for s in range(n))
    return {"numerator": int(obs_prob.numerator), "denominator": int(obs_prob.denominator)}
