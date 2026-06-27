"""Birth-death Markov chain with one transition rate missing.

States {0, 1, ..., n}. Birth rate lambda_i governs i -> i+1 (for i=0..n-1).
Death rate mu_i governs i -> i-1 (for i=1..n).

Agent A holds: all birth and death rates EXCEPT one rate r_k (either a
  lambda_i or a mu_i), presented as natural-language sentences.
Agent B holds: exactly that one missing rate r_k.

A must scan all rate sentences, find the gap in the lambda/mu sequence,
and ask B for the missing rate. After receiving it, A applies the
detailed-balance formula:
  pi_i = pi_0 * (lambda_0 * ... * lambda_{i-1}) / (mu_1 * ... * mu_i)
then normalises so sum(pi) = 1 and reports pi(target_state).

Answer: pi(target_state) as a fully reduced fraction {numerator, denominator}.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List
import random

FAMILY_NAME = "birth_death_missing_rate"
FAMILY_TYPE = "B"  # Variable hint slot: any birth rate lambda_i or death rate mu_i

_BIRTH_TEMPLATES = [
    "The birth rate from state {i} to state {j} is {v}.",
    "From state {i}, the chain moves to state {j} at rate {v}.",
    "A chain in state {i} transitions upward to state {j} with rate {v}.",
    "State {i} has an upward transition rate of {v} to state {j}.",
]

_DEATH_TEMPLATES = [
    "The death rate from state {i} to state {j} is {v}.",
    "From state {i}, the chain moves down to state {j} at rate {v}.",
    "A chain in state {i} transitions downward to state {j} with rate {v}.",
    "State {i} has a downward transition rate of {v} to state {j}.",
]


def _nl_birth(rng: random.Random, i: int, v: int) -> str:
    return rng.choice(_BIRTH_TEMPLATES).format(i=i, j=i + 1, v=v)


def _nl_death(rng: random.Random, i: int, v: int) -> str:
    return rng.choice(_DEATH_TEMPLATES).format(i=i, j=i - 1, v=v)


def _steady_state(lambdas: List[Fraction], mus: List[Fraction], n_states: int) -> List[Fraction]:
    """Compute steady-state via detailed balance.

    n_states = number of states (states 0..n_states-1).
    lambdas[i] = lambda_i for i in 0..n_states-2  (birth rate from i to i+1)
    mus[i]     = mu_i    for i in 1..n_states-1   (death rate from i to i-1),
                 stored as mus[0..n_states-2] corresponding to mu_1..mu_{n_states-1}.
    """
    # pi[i] = pi[0] * product(lambda[k]/mu[k+1] for k in 0..i-1)
    pi = [Fraction(1)]  # pi[0] = 1 (unnormalised)
    for i in range(1, n_states):
        pi.append(pi[-1] * lambdas[i - 1] / mus[i - 1])
    total = sum(pi)
    return [p / total for p in pi]


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    n_states = difficulty + 2   # 3, 4, 5 states for difficulty 1, 2, 3
    max_rate = 3 + 2 * difficulty  # 5, 7, 9

    while True:
        # Generate positive integer birth and death rates
        lambdas = [rng.randint(1, max_rate) for _ in range(n_states - 1)]
        mus = [rng.randint(1, max_rate) for _ in range(n_states - 1)]

        # Decide which rate is missing: 0..n_states-2 -> lambda, n_states-1..2*(n_states-1)-1 -> mu
        n_rates = 2 * (n_states - 1)
        missing_idx = rng.randint(0, n_rates - 1)

        # Determine if it's a lambda or mu and which index
        if missing_idx < n_states - 1:
            missing_kind = "lambda"
            missing_state = missing_idx   # lambda_missing_state
        else:
            missing_kind = "mu"
            missing_state = missing_idx - (n_states - 1) + 1  # mu_{missing_state} (1-indexed)

        target_state = rng.randint(0, n_states - 1)

        # Compute steady state
        try:
            pi = _steady_state(
                [Fraction(l) for l in lambdas],
                [Fraction(m) for m in mus],
                n_states,
            )
        except ZeroDivisionError:
            continue

        answer = pi[target_state]
        if answer.denominator > 300:
            continue
        if answer <= 0:
            continue

        break

    # Build NL constraints
    # A knows all rates except the missing one
    a_text: List[str] = []
    a_machine: List[Dict[str, Any]] = []

    for i in range(n_states - 1):
        # lambda_i: birth rate from state i to i+1
        if missing_kind == "lambda" and i == missing_state:
            continue
        v = lambdas[i]
        a_text.append(_nl_birth(rng, i, v))
        a_machine.append({"type": "birth_rate", "from_state": i, "to_state": i + 1, "value": v})

    for idx in range(n_states - 1):
        # mu_{idx+1}: death rate from state idx+1 to idx
        state = idx + 1
        if missing_kind == "mu" and state == missing_state:
            continue
        v = mus[idx]
        a_text.append(_nl_death(rng, state, v))
        a_machine.append({"type": "death_rate", "from_state": state, "to_state": state - 1, "value": v})

    # B knows the missing rate
    if missing_kind == "lambda":
        hint_val = lambdas[missing_state]
        b_text = [f"lambda_{missing_state} = {hint_val}"]
        b_machine = [{"type": "birth_rate", "from_state": missing_state,
                      "to_state": missing_state + 1, "value": hint_val}]
        hint_kind = "birth_rate"
        hint_from = missing_state
        hint_to = missing_state + 1
    else:
        idx = missing_state - 1   # mus index
        hint_val = mus[idx]
        b_text = [f"mu_{missing_state} = {hint_val}"]
        b_machine = [{"type": "death_rate", "from_state": missing_state,
                      "to_state": missing_state - 1, "value": hint_val}]
        hint_kind = "death_rate"
        hint_from = missing_state
        hint_to = missing_state - 1

    return {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "rational",
        "global_solution": {
            "numerator": int(answer.numerator),
            "denominator": int(answer.denominator),
        },
        "global_metadata": {
            "n_states": n_states,
            "lambdas": lambdas,
            "mus": mus,
            "missing_kind": missing_kind,
            "missing_state": missing_state,
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
                    "kind": hint_kind,
                    "from_state": hint_from,
                    "to_state": hint_to,
                    "value": hint_val,
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }


def _known_rates(view: Dict[str, Any]) -> set:
    """Return set of (type, from_state) pairs that are known."""
    result = set()
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] in ("birth_rate", "death_rate"):
            result.add((c["type"], c["from_state"]))
    return result


def _all_rates_known(view: Dict[str, Any], n_states: int) -> bool:
    known = _known_rates(view)
    for i in range(n_states - 1):
        if ("birth_rate", i) not in known:
            return False
    for s in range(1, n_states):
        if ("death_rate", s) not in known:
            return False
    return True


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_states"]
    return not _all_rates_known(agent_view, n)


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    kind = hint["kind"]
    if kind in ("birth_rate", "death_rate"):
        fs, ts, v = hint["from_state"], hint["to_state"], hint["value"]
        new_view["private_data"]["constraints_machine"].append(
            {"type": kind, "from_state": fs, "to_state": ts, "value": v}
        )
        rate_name = "lambda" if kind == "birth_rate" else "mu"
        new_view["private_data"]["constraints_text"].append(
            f"{rate_name}_{fs} = {v}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_states"]
    return _all_rates_known(agent_view, n)


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    n_states = instance["global_metadata"]["n_states"]
    lambdas = [Fraction(l) for l in instance["global_metadata"]["lambdas"]]
    mus = [Fraction(m) for m in instance["global_metadata"]["mus"]]
    target = instance["global_metadata"]["target_state"]
    pi = _steady_state(lambdas, mus, n_states)
    result = pi[target]
    return {"numerator": int(result.numerator), "denominator": int(result.denominator)}
