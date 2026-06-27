"""Dataset validator CLI for mira_math."""
from __future__ import annotations

from typing import Any, Dict, List
import argparse

from mira_math.schema import load_jsonl, basic_schema_checks

from mira_math.families.linear import (
    FAMILY_NAME as LINEAR_FAMILY,
    is_locally_illposed as linear_ill,
    is_uniquely_solvable as linear_unique,
    apply_hint as linear_apply,
    solve_global as linear_solve,
)
from mira_math.families.crt import (
    FAMILY_NAME as CRT_FAMILY,
    is_locally_illposed as crt_ill,
    is_uniquely_solvable as crt_unique,
    apply_hint as crt_apply,
    solve_global as crt_solve,
)
from mira_math.families.poly_interp import (
    FAMILY_NAME as POLY_FAMILY,
    is_locally_illposed as poly_ill,
    is_uniquely_solvable as poly_unique,
    apply_hint as poly_apply,
    solve_global as poly_solve,
)
from mira_math.families.recurrence import (
    FAMILY_NAME as REC_FAMILY,
    is_locally_illposed as rec_ill,
    is_uniquely_solvable as rec_unique,
    apply_hint as rec_apply,
    solve_global as rec_solve,
)
from mira_math.families.rankdef_linear_shared import (
    FAMILY_NAME as RANKDEF_FAMILY,
    is_locally_illposed as rankdef_ill,
    is_uniquely_solvable as rankdef_unique,
    apply_hint as rankdef_apply,
    solve_global as rankdef_solve,
)
from mira_math.families.laplace_grid import (
    FAMILY_NAME as LAPLACE_FAMILY,
    is_locally_illposed as laplace_ill,
    is_uniquely_solvable as laplace_unique,
    apply_hint as laplace_apply,
    solve_global as laplace_solve,
)
from mira_math.families.deconvolution import (
    FAMILY_NAME as DECONV_FAMILY,
    is_locally_illposed as deconv_ill,
    is_uniquely_solvable as deconv_unique,
    apply_hint as deconv_apply,
    solve_global as deconv_solve,
)
from mira_math.families.phase_retrieval import (
    FAMILY_NAME as PHASE_FAMILY,
    is_locally_illposed as phase_ill,
    is_uniquely_solvable as phase_unique,
    apply_hint as phase_apply,
    solve_global as phase_solve,
)
from mira_math.families.matrix_completion import (
    FAMILY_NAME as MATRIX_FAMILY,
    is_locally_illposed as matrix_ill,
    is_uniquely_solvable as matrix_unique,
    apply_hint as matrix_apply,
    solve_global as matrix_solve,
)
from mira_math.families.tomography import (
    FAMILY_NAME as TOMO_FAMILY,
    is_locally_illposed as tomo_ill,
    is_uniquely_solvable as tomo_unique,
    apply_hint as tomo_apply,
    solve_global as tomo_solve,
)
from mira_math.families.graph_path_sums import (
    FAMILY_NAME as GRAPH_FAMILY,
    is_locally_illposed as graph_ill,
    is_uniquely_solvable as graph_unique,
    apply_hint as graph_apply,
    solve_global as graph_solve,
)
from mira_math.families.moment_problem import (
    FAMILY_NAME as MOMENT_FAMILY,
    is_locally_illposed as moment_ill,
    is_uniquely_solvable as moment_unique,
    apply_hint as moment_apply,
    solve_global as moment_solve,
)
from mira_math.families.geometry_coordinates import (
    FAMILY_NAME as GEO_FAMILY,
    is_locally_illposed as geo_ill,
    is_uniquely_solvable as geo_unique,
    apply_hint as geo_apply,
    solve_global as geo_solve,
)
from mira_math.families.bayes_missing_prior import (
    FAMILY_NAME as BAYES_FAMILY,
    is_locally_illposed as bayes_ill,
    is_uniquely_solvable as bayes_unique,
    apply_hint as bayes_apply,
    solve_global as bayes_solve,
)
from mira_math.families.piecewise_missing_threshold import (
    FAMILY_NAME as PIECEWISE_FAMILY,
    is_locally_illposed as piecewise_ill,
    is_uniquely_solvable as piecewise_unique,
    apply_hint as piecewise_apply,
    solve_global as piecewise_solve,
)
from mira_math.families.markov_missing_transition import (
    FAMILY_NAME as MARKOV_FAMILY,
    is_locally_illposed as markov_ill,
    is_uniquely_solvable as markov_unique,
    apply_hint as markov_apply,
    solve_global as markov_solve,
)
from mira_math.families.linear_system_missing_coeff import (
    FAMILY_NAME as LINSYS_FAMILY,
    is_locally_illposed as linsys_ill,
    is_uniquely_solvable as linsys_unique,
    apply_hint as linsys_apply,
    solve_global as linsys_solve,
)
from mira_math.families.circuit_missing_resistance import (
    FAMILY_NAME as CIRCUIT_FAMILY,
    is_locally_illposed as circuit_ill,
    is_uniquely_solvable as circuit_unique,
    apply_hint as circuit_apply,
    solve_global as circuit_solve,
)
from mira_math.families.portfolio_variance_missing_corr import (
    FAMILY_NAME as PORTFOLIO_FAMILY,
    is_locally_illposed as portfolio_ill,
    is_uniquely_solvable as portfolio_unique,
    apply_hint as portfolio_apply,
    solve_global as portfolio_solve,
)
from mira_math.families.birth_death_missing_rate import (
    FAMILY_NAME as BIRTHDEATH_FAMILY,
    is_locally_illposed as birthdeath_ill,
    is_uniquely_solvable as birthdeath_unique,
    apply_hint as birthdeath_apply,
    solve_global as birthdeath_solve,
)
from mira_math.families.eigenvector_missing_entry import (
    FAMILY_NAME as EIGVEC_FAMILY,
    is_locally_illposed as eigvec_ill,
    is_uniquely_solvable as eigvec_unique,
    apply_hint as eigvec_apply,
    solve_global as eigvec_solve,
)
from mira_math.families.steady_state_missing_emission import (
    FAMILY_NAME as SSEMIT_FAMILY,
    is_locally_illposed as ssemit_ill,
    is_uniquely_solvable as ssemit_unique,
    apply_hint as ssemit_apply,
    solve_global as ssemit_solve,
)


FAMILY_VALIDATORS = {
    LINEAR_FAMILY: (linear_ill, linear_unique, linear_apply, linear_solve),
    CRT_FAMILY: (crt_ill, crt_unique, crt_apply, crt_solve),
    POLY_FAMILY: (poly_ill, poly_unique, poly_apply, poly_solve),
    REC_FAMILY: (rec_ill, rec_unique, rec_apply, rec_solve),
    RANKDEF_FAMILY: (rankdef_ill, rankdef_unique, rankdef_apply, rankdef_solve),
    LAPLACE_FAMILY: (laplace_ill, laplace_unique, laplace_apply, laplace_solve),
    DECONV_FAMILY: (deconv_ill, deconv_unique, deconv_apply, deconv_solve),
    PHASE_FAMILY: (phase_ill, phase_unique, phase_apply, phase_solve),
    MATRIX_FAMILY: (matrix_ill, matrix_unique, matrix_apply, matrix_solve),
    TOMO_FAMILY: (tomo_ill, tomo_unique, tomo_apply, tomo_solve),
    GRAPH_FAMILY: (graph_ill, graph_unique, graph_apply, graph_solve),
    MOMENT_FAMILY: (moment_ill, moment_unique, moment_apply, moment_solve),
    GEO_FAMILY: (geo_ill, geo_unique, geo_apply, geo_solve),
    BAYES_FAMILY: (bayes_ill, bayes_unique, bayes_apply, bayes_solve),
    PIECEWISE_FAMILY: (piecewise_ill, piecewise_unique, piecewise_apply, piecewise_solve),
    MARKOV_FAMILY: (markov_ill, markov_unique, markov_apply, markov_solve),
    LINSYS_FAMILY: (linsys_ill, linsys_unique, linsys_apply, linsys_solve),
    CIRCUIT_FAMILY: (circuit_ill, circuit_unique, circuit_apply, circuit_solve),
    PORTFOLIO_FAMILY: (portfolio_ill, portfolio_unique, portfolio_apply, portfolio_solve),
    BIRTHDEATH_FAMILY: (birthdeath_ill, birthdeath_unique, birthdeath_apply, birthdeath_solve),
    EIGVEC_FAMILY: (eigvec_ill, eigvec_unique, eigvec_apply, eigvec_solve),
    SSEMIT_FAMILY: (ssemit_ill, ssemit_unique, ssemit_apply, ssemit_solve),
}


def _get_view(inst: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    for v in inst["agent_views"]:
        if v["agent_id"] == agent_id:
            return v
    raise KeyError(agent_id)


def validate_instance(inst: Dict[str, Any]) -> List[str]:
    errors = basic_schema_checks(inst)
    family = inst.get("family")
    if family not in FAMILY_VALIDATORS:
        errors.append(f"unknown family: {family}")
        return errors

    is_ill, is_unique, apply_hint, solve_global = FAMILY_VALIDATORS[family]

    # Global uniqueness and correctness
    try:
        solved = solve_global(inst)
        if solved != inst["global_solution"]:
            errors.append("global_solution does not match solver")
    except Exception as e:
        errors.append(f"global solver failed: {e}")

    # Local ill-posedness
    for v in inst["agent_views"]:
        try:
            if not is_ill(inst, v):
                errors.append(f"agent {v['agent_id']} view appears well-posed")
        except Exception as e:
            errors.append(f"local check failed for {v['agent_id']}: {e}")

    # Minimal hint sufficiency
    for hint in inst["minimal_hint_spec"]["atomic_hints"]:
        for consumer in hint.get("consumers", []):
            try:
                view = _get_view(inst, consumer)
                view_with_hint = apply_hint(view, hint)
                if not is_unique(inst, view_with_hint):
                    errors.append(f"hint {hint.get('hint_id')} does not resolve for {consumer}")
            except Exception as e:
                errors.append(f"hint application failed: {e}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    args = parser.parse_args()

    instances = load_jsonl(args.input_path)
    total_errors = 0
    for inst in instances:
        errs = validate_instance(inst)
        if errs:
            total_errors += len(errs)
            print(f"{inst.get('id','?')}: {errs}")

    if total_errors:
        raise SystemExit(1)

    print(f"validated {len(instances)} instances")


if __name__ == "__main__":
    main()
