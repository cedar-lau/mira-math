"""Portfolio variance with one pairwise correlation missing.

n assets, each with weight w_i, standard deviation sigma_i, and pairwise
correlations rho(i,j) for all i < j.

Portfolio variance: sigma_p^2 = sum_{i,j} w_i * w_j * sigma_i * sigma_j * rho(i,j)
where rho(i,i) = 1.

Agent A holds: all weights, all standard deviations, all pairwise correlations
  EXCEPT one rho(a,b) (randomised pair), presented as natural-language sentences.
Agent B holds: exactly rho(a,b).

A must check all n*(n-1)/2 correlation pairs to find the missing one, then ask
B for it. After receiving it, A computes sigma_p^2 and reports as a fraction.

Answer: sigma_p^2 as a fully reduced fraction {numerator, denominator}.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Tuple
import random

FAMILY_NAME = "portfolio_variance_missing_corr"
FAMILY_TYPE = "B"  # Variable hint slot: any pairwise correlation rho(i,j)

_WEIGHT_TEMPLATES = [
    "Asset {i} has portfolio weight {w}.",
    "The weight of asset {i} in the portfolio is {w}.",
    "Asset {i} is allocated a fraction {w} of the portfolio.",
]

_STD_TEMPLATES = [
    "Asset {i} has standard deviation {s}.",
    "The standard deviation of asset {i} is {s}.",
    "Asset {i}'s volatility (standard deviation) is {s}.",
]

_CORR_TEMPLATES = [
    "The correlation between asset {i} and asset {j} is {r}.",
    "rho({i},{j}) = {r}.",
    "Assets {i} and {j} have a pairwise correlation of {r}.",
    "The correlation coefficient rho({i},{j}) equals {r}.",
]


def _frac_text(f: Fraction) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _nl_weight(rng: random.Random, i: int, w: Fraction) -> str:
    return rng.choice(_WEIGHT_TEMPLATES).format(i=i, w=_frac_text(w))


def _nl_std(rng: random.Random, i: int, s: int) -> str:
    return rng.choice(_STD_TEMPLATES).format(i=i, s=s)


def _nl_corr(rng: random.Random, i: int, j: int, r: Fraction) -> str:
    return rng.choice(_CORR_TEMPLATES).format(i=i, j=j, r=_frac_text(r))


def _portfolio_variance(
    weights: List[Fraction],
    sigmas: List[int],
    corrs: Dict[Tuple[int, int], Fraction],
    n: int,
) -> Fraction:
    total = Fraction(0)
    for i in range(n):
        for j in range(n):
            rho = corrs.get((min(i, j), max(i, j)), Fraction(1) if i == j else None)
            if rho is None:
                raise ValueError(f"Missing correlation rho({i},{j})")
            total += weights[i] * weights[j] * sigmas[i] * sigmas[j] * rho
    return total


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    n = difficulty + 1   # 2, 3, 4 assets

    while True:
        # Weights: random positive integers normalised to sum to 1
        parts = [rng.randint(1, 4) for _ in range(n)]
        total_parts = sum(parts)
        weights = [Fraction(p, total_parts) for p in parts]

        # Standard deviations: small positive integers
        sigmas = [rng.randint(1, 5) for _ in range(n)]

        # Correlations for all i < j: rational in (-1, 1) with small denominator
        # Use multiples of 1/d for d in {2, 3, 4}
        corrs: Dict[Tuple[int, int], Fraction] = {}
        for i in range(n):
            for j in range(i + 1, n):
                d = rng.choice([2, 3, 4])
                num = rng.randint(-(d - 1), d - 1)
                corrs[(i, j)] = Fraction(num, d)

        # Pick missing pair
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        missing_pair = rng.choice(pairs)

        # Compute answer
        try:
            var = _portfolio_variance(weights, sigmas, corrs, n)
        except ValueError:
            continue

        if var.denominator > 500:
            continue
        if var <= 0:
            continue

        target_state = rng.randint(0, n - 1)  # unused but kept for consistency
        break

    # Build constraints
    a_text: List[str] = []
    a_machine: List[Dict[str, Any]] = []

    for i in range(n):
        a_text.append(_nl_weight(rng, i, weights[i]))
        a_machine.append({"type": "weight", "asset": i,
                          "num": int(weights[i].numerator), "denom": int(weights[i].denominator)})

    for i in range(n):
        a_text.append(_nl_std(rng, i, sigmas[i]))
        a_machine.append({"type": "std", "asset": i, "value": sigmas[i]})

    for (i, j), rho in corrs.items():
        if (i, j) == missing_pair:
            continue
        a_text.append(_nl_corr(rng, i, j, rho))
        a_machine.append({"type": "corr", "asset_i": i, "asset_j": j,
                          "num": int(rho.numerator), "denom": int(rho.denominator)})

    # B: just the missing correlation
    rho_hint = corrs[missing_pair]
    mi, mj = missing_pair
    b_text = [f"rho({mi},{mj}) = {_frac_text(rho_hint)}"]
    b_machine = [{"type": "corr", "asset_i": mi, "asset_j": mj,
                  "num": int(rho_hint.numerator), "denom": int(rho_hint.denominator)}]

    return {
        "id": instance_id,
        "family": FAMILY_NAME,
        "difficulty": difficulty,
        "n_agents": 2,
        "answer_type": "rational",
        "global_solution": {
            "numerator": int(var.numerator),
            "denominator": int(var.denominator),
        },
        "global_metadata": {
            "n_assets": n,
            "weights_num": [int(w.numerator) for w in weights],
            "weights_denom": [int(w.denominator) for w in weights],
            "sigmas": sigmas,
            "corrs": {f"{i},{j}": [int(v.numerator), int(v.denominator)]
                      for (i, j), v in corrs.items()},
            "missing_pair": list(missing_pair),
        },
        "agent_views": [
            {
                "agent_id": "A",
                "private_data": {
                    "unknowns": ["sigma_p^2"],
                    "externals": [],
                    "constraints_text": a_text,
                    "constraints_machine": a_machine,
                },
            },
            {
                "agent_id": "B",
                "private_data": {
                    "unknowns": ["sigma_p^2"],
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
                    "kind": "corr",
                    "asset_i": mi,
                    "asset_j": mj,
                    "num": int(rho_hint.numerator),
                    "denom": int(rho_hint.denominator),
                    "providers": ["B"],
                    "consumers": ["A"],
                }
            ],
        },
    }


def _known_corr_pairs(view: Dict[str, Any]) -> set:
    return {
        (c["asset_i"], c["asset_j"])
        for c in view["private_data"]["constraints_machine"]
        if c["type"] == "corr"
    }


def _has_all_data(view: Dict[str, Any], n: int) -> bool:
    known_corrs = _known_corr_pairs(view)
    all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    if not all_pairs.issubset(known_corrs):
        return False
    machine = view["private_data"]["constraints_machine"]
    known_weights = {c["asset"] for c in machine if c["type"] == "weight"}
    known_stds = {c["asset"] for c in machine if c["type"] == "std"}
    return (all(i in known_weights for i in range(n)) and
            all(i in known_stds for i in range(n)))


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_assets"]
    return not _has_all_data(agent_view, n)


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "corr":
        ai, aj = hint["asset_i"], hint["asset_j"]
        rho = Fraction(hint["num"], hint["denom"])
        new_view["private_data"]["constraints_machine"].append(
            {"type": "corr", "asset_i": ai, "asset_j": aj,
             "num": hint["num"], "denom": hint["denom"]}
        )
        new_view["private_data"]["constraints_text"].append(
            f"rho({ai},{aj}) = {_frac_text(rho)}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = instance["global_metadata"]["n_assets"]
    return _has_all_data(agent_view, n)


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    n = instance["global_metadata"]["n_assets"]
    weights = [Fraction(instance["global_metadata"]["weights_num"][i],
                        instance["global_metadata"]["weights_denom"][i])
               for i in range(n)]
    sigmas = instance["global_metadata"]["sigmas"]
    corrs: Dict[Tuple[int, int], Fraction] = {}
    for key, (num, denom) in instance["global_metadata"]["corrs"].items():
        i, j = map(int, key.split(","))
        corrs[(i, j)] = Fraction(num, denom)
    var = _portfolio_variance(weights, sigmas, corrs, n)
    return {"numerator": int(var.numerator), "denominator": int(var.denominator)}
