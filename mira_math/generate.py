"""Dataset generator CLI for mira_math."""
from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Tuple

from mira_math.schema import dump_jsonl
from mira_math.families import FAMILY_TYPES, TYPE_A_FAMILIES, TYPE_B_FAMILIES
from mira_math.families.linear import generate_instance as gen_linear, FAMILY_NAME as LINEAR_FAMILY
from mira_math.families.crt import generate_instance as gen_crt, FAMILY_NAME as CRT_FAMILY
from mira_math.families.poly_interp import generate_instance as gen_poly, FAMILY_NAME as POLY_FAMILY
from mira_math.families.recurrence import generate_instance as gen_rec, FAMILY_NAME as REC_FAMILY
from mira_math.families.rankdef_linear_shared import (
    generate_instance as gen_rankdef,
    FAMILY_NAME as RANKDEF_FAMILY,
)
from mira_math.families.laplace_grid import generate_instance as gen_laplace, FAMILY_NAME as LAPLACE_FAMILY
from mira_math.families.deconvolution import generate_instance as gen_deconv, FAMILY_NAME as DECONV_FAMILY
from mira_math.families.phase_retrieval import generate_instance as gen_phase, FAMILY_NAME as PHASE_FAMILY
from mira_math.families.matrix_completion import generate_instance as gen_matrix, FAMILY_NAME as MATRIX_FAMILY
from mira_math.families.tomography import generate_instance as gen_tomo, FAMILY_NAME as TOMO_FAMILY
from mira_math.families.graph_path_sums import generate_instance as gen_graph, FAMILY_NAME as GRAPH_FAMILY
from mira_math.families.moment_problem import generate_instance as gen_moment, FAMILY_NAME as MOMENT_FAMILY
from mira_math.families.geometry_coordinates import generate_instance as gen_geo, FAMILY_NAME as GEO_FAMILY
from mira_math.families.bayes_missing_prior import generate_instance as gen_bayes, FAMILY_NAME as BAYES_FAMILY
from mira_math.families.piecewise_missing_threshold import generate_instance as gen_piecewise, FAMILY_NAME as PIECEWISE_FAMILY
from mira_math.families.markov_missing_transition import generate_instance as gen_markov, FAMILY_NAME as MARKOV_FAMILY
from mira_math.families.linear_system_missing_coeff import generate_instance as gen_linsys, FAMILY_NAME as LINSYS_FAMILY
from mira_math.families.circuit_missing_resistance import generate_instance as gen_circuit, FAMILY_NAME as CIRCUIT_FAMILY
from mira_math.families.portfolio_variance_missing_corr import generate_instance as gen_portfolio, FAMILY_NAME as PORTFOLIO_FAMILY
from mira_math.families.birth_death_missing_rate import generate_instance as gen_birthdeath, FAMILY_NAME as BIRTHDEATH_FAMILY
from mira_math.families.eigenvector_missing_entry import generate_instance as gen_eigvec, FAMILY_NAME as EIGVEC_FAMILY
from mira_math.families.steady_state_missing_emission import generate_instance as gen_ssemit, FAMILY_NAME as SSEMIT_FAMILY


GEN_MAP = {
    LINEAR_FAMILY: gen_linear,
    CRT_FAMILY: gen_crt,
    POLY_FAMILY: gen_poly,
    REC_FAMILY: gen_rec,
    RANKDEF_FAMILY: gen_rankdef,
    LAPLACE_FAMILY: gen_laplace,
    DECONV_FAMILY: gen_deconv,
    PHASE_FAMILY: gen_phase,
    MATRIX_FAMILY: gen_matrix,
    TOMO_FAMILY: gen_tomo,
    GRAPH_FAMILY: gen_graph,
    MOMENT_FAMILY: gen_moment,
    GEO_FAMILY: gen_geo,
    BAYES_FAMILY: gen_bayes,
    PIECEWISE_FAMILY: gen_piecewise,
    MARKOV_FAMILY: gen_markov,
    LINSYS_FAMILY: gen_linsys,
    CIRCUIT_FAMILY: gen_circuit,
    PORTFOLIO_FAMILY: gen_portfolio,
    BIRTHDEATH_FAMILY: gen_birthdeath,
    EIGVEC_FAMILY: gen_eigvec,
    SSEMIT_FAMILY: gen_ssemit,
}
MIX_FAMILY = "mix"
TYPE_A_FAMILY = "type_a"   # pseudo-family: mix of all Type-A families
TYPE_B_FAMILY = "type_b"   # pseudo-family: mix of all Type-B families
TYPED_FAMILY = "typed"     # pseudo-family: separate n_a/n_b per type
NON_SCALING_FAMILIES = {CRT_FAMILY, TOMO_FAMILY}
FEW_SHOT_COUNT = 4
# Dedicated seed offset so few-shot examples never collide with dataset instances.
_FEW_SHOT_SEED_OFFSET = 900_000


# ---------------------------------------------------------------------------
# Few-shot example helpers
# ---------------------------------------------------------------------------

def _format_ideal_request(instance: Dict[str, Any]) -> str:
    """Return the ideal natural-language request Agent A should make."""
    family = instance["family"]
    h = instance["minimal_hint_spec"]["atomic_hints"][0]

    if family == "linear_system_separator":
        return f"the value of variable {h['var']}"
    elif family == "crt_reconstruction":
        return "your remaining congruence"
    elif family == "poly_interpolation":
        return "your interpolation point"
    elif family == "recurrence_missing_init":
        return f"the initial value a({h.get('index', 1)})"
    elif family == "rankdef_linear_shared":
        return "your additional equation"
    elif family == "laplace_grid":
        return f"the value of {h['var']}"
    elif family == "deconvolution":
        return f"the measurement y[{h['index']}]"
    elif family == "phase_retrieval":
        return "the sign of x0 (+1 or -1)"
    elif family == "matrix_completion":
        return f"the matrix entry m[{h['row']}][{h['col']}]"
    elif family == "discrete_tomography":
        return f"the cell value at ({h['row']},{h['col']})"
    elif family == "graph_path_sums":
        return "your path sum equation"
    elif family == "moment_problem":
        return "the 2nd moment"
    elif family == "geometry_coordinates":
        return "your line equation"
    elif family == "bayes_missing_prior":
        return "the prior P(H)"
    elif family == "piecewise_missing_threshold":
        return "the threshold t"
    elif family == "markov_missing_transition":
        return f"the transition probability P({h['from_state']} -> {h['to_state']})"
    elif family == "linear_system_missing_coeff":
        return f"the coefficient A[{h['row']}][{h['col']}]"
    elif family == "birth_death_missing_rate":
        if h["kind"] == "birth_rate":
            return f"the birth rate lambda_{h['from_state']}"
        else:
            return f"the death rate mu_{h['to_state']}"
    elif family == "circuit_missing_resistance":
        return f"the resistance R{h['resistor']}"
    elif family == "eigenvector_missing_entry":
        return f"the entry M[{h['row']}][{h['col']}]"
    elif family == "portfolio_variance_missing_corr":
        return f"the correlation rho({h['asset_i']},{h['asset_j']})"
    elif family == "steady_state_missing_emission":
        return f"the emission probability e({h['state']},{h['symbol']})"
    else:
        return "the missing information"


def _format_hint_value(instance: Dict[str, Any]) -> str:
    """Return B's hint as a short string (what B would say in an offer)."""
    b_view = next(v for v in instance["agent_views"] if v["agent_id"] == "B")
    # B's first constraint text is the canonical hint phrasing
    return b_view["private_data"]["constraints_text"][0]


def _format_answer(instance: Dict[str, Any]) -> str:
    """Format the global solution as the answer string Agent A should produce."""
    return json.dumps(instance["global_solution"], sort_keys=True)


def _build_few_shot_example(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a generated instance into a compact few-shot example for Agent A."""
    a_view = next(v for v in instance["agent_views"] if v["agent_id"] == "A")
    return {
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "constraints": a_view["private_data"]["constraints_text"],
        "unknowns": a_view["private_data"].get("unknowns", []),
        "request": _format_ideal_request(instance),
        "hint": _format_hint_value(instance),
        "answer": instance["global_solution"],
        "answer_type": instance.get("answer_type", "int"),
    }


def generate_few_shot_bank(
    families: List[str],
    difficulties: List[int],
    seed: int,
) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """Generate a bank of few-shot examples: 4 per (family, difficulty).

    Uses a dedicated seed range to avoid collisions with dataset instances.
    Returns a dict keyed by (family_name, difficulty).
    """
    bank: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for family in families:
        for diff in difficulties:
            eff = _effective_difficulty(family, diff)
            rng = random.Random(seed + _FEW_SHOT_SEED_OFFSET + hash((family, diff)) % 100_000)
            examples = []
            for i in range(FEW_SHOT_COUNT):
                inst = GEN_MAP[family](
                    rng, difficulty=eff,
                    instance_id=f"fs-{family[:3]}-d{diff}-{i}",
                )
                examples.append(_build_few_shot_example(inst))
            bank[(family, diff)] = examples
    return bank


def _attach_few_shot(
    instances: List[Dict[str, Any]],
    seed: int,
) -> None:
    """Attach few-shot examples to each instance in-place."""
    # Collect the unique (family, difficulty) pairs
    pairs = {(inst["family"], inst["difficulty"]) for inst in instances}
    families = list({f for f, _ in pairs})
    difficulties = sorted({d for _, d in pairs})

    bank = generate_few_shot_bank(families, difficulties, seed)

    for inst in instances:
        key = (inst["family"], inst["difficulty"])
        inst["few_shot_examples"] = bank.get(key, [])


def _effective_difficulty(family: str, difficulty: int) -> int:
    if family in NON_SCALING_FAMILIES:
        return 1
    return difficulty


def _parse_mix_families(mix_families: str | None) -> List[str]:
    if not mix_families:
        return list(GEN_MAP.keys())
    families = [f.strip() for f in mix_families.split(",") if f.strip()]
    unknown = [f for f in families if f not in GEN_MAP]
    if unknown:
        raise ValueError(f"unknown families in mix: {unknown}")
    if not families:
        raise ValueError("mix_families must include at least one family")
    return families


def generate_instances(
    family: str,
    n: int,
    seed: int,
    difficulty: int,
    mix_families: str | None = None,
    id_offset: int = 0,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    # Resolve type_a / type_b pseudo-families into mix over their respective lists
    if family == TYPE_A_FAMILY:
        family = MIX_FAMILY
        mix_families = mix_families or ",".join(TYPE_A_FAMILIES)
    elif family == TYPE_B_FAMILY:
        family = MIX_FAMILY
        mix_families = mix_families or ",".join(TYPE_B_FAMILIES)

    if family != MIX_FAMILY:
        if family not in GEN_MAP:
            raise ValueError(
                f"unknown family: {family!r}. "
                f"Valid values: {sorted(GEN_MAP)} + mix, type_a, type_b"
            )
        instances = []
        for i in range(n):
            instance_id = f"{family[:3]}-{i + id_offset:06d}"
            eff = _effective_difficulty(family, difficulty)
            inst = GEN_MAP[family](rng, difficulty=eff, instance_id=instance_id)
            instances.append(inst)
        return instances

    families = _parse_mix_families(mix_families)
    # Distribute counts as evenly as possible across families.
    base = n // len(families)
    remainder = n % len(families)
    family_counts = {f: base for f in families}
    # Use RNG to assign the remainder for a deterministic but shuffled mix.
    remainder_families = families[:]
    rng.shuffle(remainder_families)
    for f in remainder_families[:remainder]:
        family_counts[f] += 1

    family_sequence: List[str] = []
    for f in families:
        family_sequence.extend([f] * family_counts[f])
    rng.shuffle(family_sequence)

    instances = []
    for i, fam in enumerate(family_sequence):
        instance_id = f"{fam[:3]}-{i + id_offset:06d}"
        eff = _effective_difficulty(fam, difficulty)
        inst = GEN_MAP[fam](rng, difficulty=eff, instance_id=instance_id)
        instances.append(inst)
    return instances


def generate_typed_instances(
    n_a: int,
    n_b: int,
    seed: int,
    difficulties: List[int],
) -> List[Dict[str, Any]]:
    """Generate n_a instances per difficulty per Type-A family,
    and n_b instances per difficulty per Type-B family.

    Example: n_a=20, n_b=50, difficulties=[1,2,3]
      -> 20*3*11 = 660 Type-A instances + 50*3*11 = 1650 Type-B instances = 2310 total
    """
    instances: List[Dict[str, Any]] = []
    gid = 0

    for ftype, families, n_per_diff in [
        ("A", sorted(TYPE_A_FAMILIES), n_a),
        ("B", sorted(TYPE_B_FAMILIES), n_b),
    ]:
        for family in families:
            rng = random.Random(seed)
            for diff in difficulties:
                eff = _effective_difficulty(family, diff)
                for _ in range(n_per_diff):
                    inst = GEN_MAP[family](
                        rng, difficulty=eff,
                        instance_id=f"{family[:3]}-{gid:06d}",
                    )
                    gid += 1
                    instances.append(inst)

    return instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        required=True,
        help=(
            "Family name, 'mix' (all families), "
            "'type_a' (all Type-A / fixed-slot families), "
            "'type_b' (all Type-B / variable-slot families), "
            "or 'typed' (separate counts per type via --n-a / --n-b)."
        ),
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Instances per difficulty per family (required unless --family typed).")
    parser.add_argument("--n-a", type=int, default=None,
                        help="For --family typed: instances per difficulty per Type-A family.")
    parser.add_argument("--n-b", type=int, default=None,
                        help="For --family typed: instances per difficulty per Type-B family.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument(
        "--difficulties",
        default=None,
        help="Comma-separated list of difficulties; generates n instances per difficulty.",
    )
    parser.add_argument(
        "--mix_families",
        default=None,
        help="Comma-separated families for --family mix (default: all).",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--few-shot",
        action="store_true",
        default=False,
        help="Attach 4 worked examples per (family, difficulty) to each instance for few-shot prompting.",
    )
    args = parser.parse_args()

    difficulties: List[int]
    if args.difficulties:
        difficulties = [int(d.strip()) for d in args.difficulties.split(",") if d.strip()]
        if not difficulties:
            raise ValueError("difficulties must include at least one integer")
    else:
        difficulties = [args.difficulty]

    if args.family == TYPED_FAMILY:
        if args.n_a is None or args.n_b is None:
            parser.error("--family typed requires both --n-a and --n-b")
        instances = generate_typed_instances(
            n_a=args.n_a,
            n_b=args.n_b,
            seed=args.seed,
            difficulties=difficulties,
        )
    else:
        if args.n is None:
            parser.error("--n is required unless --family typed")
        instances: List[Dict[str, Any]] = []
        offset = 0
        for diff in difficulties:
            instances.extend(
                generate_instances(
                    args.family,
                    args.n,
                    args.seed,
                    diff,
                    mix_families=args.mix_families,
                    id_offset=offset,
                )
            )
            offset += args.n
    if args.few_shot:
        _attach_few_shot(instances, args.seed)

    dump_jsonl(args.out, instances)


if __name__ == "__main__":
    main()
