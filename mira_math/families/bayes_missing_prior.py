"""Bayesian inference with missing prior (base rate).

Agent A holds the likelihoods P(E|H) and P(E|¬H).
Agent B holds the prior P(H).
Neither can compute the posterior alone; together they apply Bayes' theorem.
Answer: P(H|E) as a reduced fraction {numerator, denominator}.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List
import random


FAMILY_NAME = "bayes_missing_prior"
FAMILY_TYPE = "A"  # Fixed hint slot: always the prior P(H)


def _frac_text(num: int, den: int) -> str:
    """Return 'num/den' or just 'num' when den==1."""
    if den == 1:
        return str(num)
    return f"{num}/{den}"


def _posterior(sens: Fraction, fpr: Fraction, prior: Fraction) -> Fraction:
    p_e = sens * prior + fpr * (1 - prior)
    return (sens * prior) / p_e


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    max_denom = 3 + 2 * difficulty  # 5 at d=1, 7 at d=2, 9 at d=3

    while True:
        prior_denom = rng.randint(2, max_denom)
        prior_num = rng.randint(1, prior_denom - 1)
        prior = Fraction(prior_num, prior_denom)

        sens_denom = rng.randint(2, max_denom)
        sens_num = rng.randint(1, sens_denom - 1)
        sens = Fraction(sens_num, sens_denom)

        fpr_denom = rng.randint(2, max_denom)
        fpr_num = rng.randint(1, fpr_denom - 1)
        fpr = Fraction(fpr_num, fpr_denom)

        # Test must be informative (sensitivity ≠ false-positive rate)
        if sens == fpr:
            continue

        p_e = sens * prior + fpr * (1 - prior)
        if p_e == 0:
            continue

        post = _posterior(sens, fpr, prior)

        # Keep the posterior denominator manageable for hand/LLM computation
        if post.denominator > 200:
            continue

        break

    sn, sd = int(sens.numerator), int(sens.denominator)
    fn, fd = int(fpr.numerator), int(fpr.denominator)
    pn, pd = int(prior.numerator), int(prior.denominator)

    instance = {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "dict_int",
        "global_solution": {
            "numerator": int(post.numerator),
            "denominator": int(post.denominator),
        },
        "global_metadata": {
            "sens_num": sn, "sens_denom": sd,
            "fpr_num": fn, "fpr_denom": fd,
            "prior_num": pn, "prior_denom": pd,
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": ["P_H_given_E"],
                    "externals": [],
                    "constraints_text": [
                        f"P(E|H) = {_frac_text(sn, sd)}",
                        f"P(E|not_H) = {_frac_text(fn, fd)}",
                    ],
                    "constraints_machine": [
                        {"type": "likelihood_pos", "num": sn, "denom": sd},
                        {"type": "likelihood_neg", "num": fn, "denom": fd},
                    ],
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": ["P_H_given_E"],
                    "externals": [],
                    "constraints_text": [
                        f"P(H) = {_frac_text(pn, pd)}",
                    ],
                    "constraints_machine": [
                        {"type": "prior", "num": pn, "denom": pd},
                    ],
                },
            },
        ],
        "minimal_hint_spec": {
            "k_min": 1,
            "atomic_hints": [
                {
                    "hint_id": "h1",
                    "kind": "prior",
                    "num": pn,
                    "denom": pd,
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }
    return instance


def _has_type(view: Dict[str, Any], ctype: str) -> bool:
    return any(c["type"] == ctype for c in view["private_data"]["constraints_machine"])


def _is_solvable(view: Dict[str, Any]) -> bool:
    return (
        _has_type(view, "likelihood_pos")
        and _has_type(view, "likelihood_neg")
        and _has_type(view, "prior")
    )


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    return not _is_solvable(agent_view)


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
    num, den = hint["num"], hint["denom"]
    if kind == "prior":
        label = f"P(H) = {_frac_text(num, den)}"
        ctype = "prior"
    elif kind == "likelihood_pos":
        label = f"P(E|H) = {_frac_text(num, den)}"
        ctype = "likelihood_pos"
    elif kind == "likelihood_neg":
        label = f"P(E|not_H) = {_frac_text(num, den)}"
        ctype = "likelihood_neg"
    else:
        return new_view
    new_view["private_data"]["constraints_machine"].append({"type": ctype, "num": num, "denom": den})
    new_view["private_data"]["constraints_text"].append(label)
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    return _is_solvable(agent_view)


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    sens = fpr = prior = None
    for v in instance["agent_views"]:
        for c in v["private_data"]["constraints_machine"]:
            if c["type"] == "likelihood_pos":
                sens = Fraction(c["num"], c["denom"])
            elif c["type"] == "likelihood_neg":
                fpr = Fraction(c["num"], c["denom"])
            elif c["type"] == "prior":
                prior = Fraction(c["num"], c["denom"])
    if sens is None or fpr is None or prior is None:
        raise ValueError("missing Bayes data across agent views")
    post = _posterior(sens, fpr, prior)
    return {"numerator": int(post.numerator), "denominator": int(post.denominator)}