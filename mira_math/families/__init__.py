"""Problem family registry."""

from mira_math.families.linear import FAMILY_NAME as LINEAR_FAMILY, FAMILY_TYPE as LINEAR_TYPE
from mira_math.families.crt import FAMILY_NAME as CRT_FAMILY, FAMILY_TYPE as CRT_TYPE
from mira_math.families.poly_interp import FAMILY_NAME as POLY_FAMILY, FAMILY_TYPE as POLY_TYPE
from mira_math.families.recurrence import FAMILY_NAME as REC_FAMILY, FAMILY_TYPE as REC_TYPE
from mira_math.families.rankdef_linear_shared import FAMILY_NAME as RANKDEF_FAMILY, FAMILY_TYPE as RANKDEF_TYPE
from mira_math.families.laplace_grid import FAMILY_NAME as LAPLACE_FAMILY, FAMILY_TYPE as LAPLACE_TYPE
from mira_math.families.deconvolution import FAMILY_NAME as DECONV_FAMILY, FAMILY_TYPE as DECONV_TYPE
from mira_math.families.phase_retrieval import FAMILY_NAME as PHASE_FAMILY, FAMILY_TYPE as PHASE_TYPE
from mira_math.families.matrix_completion import FAMILY_NAME as MATRIX_FAMILY, FAMILY_TYPE as MATRIX_TYPE
from mira_math.families.tomography import FAMILY_NAME as TOMO_FAMILY, FAMILY_TYPE as TOMO_TYPE
from mira_math.families.graph_path_sums import FAMILY_NAME as GRAPH_FAMILY, FAMILY_TYPE as GRAPH_TYPE
from mira_math.families.moment_problem import FAMILY_NAME as MOMENT_FAMILY, FAMILY_TYPE as MOMENT_TYPE
from mira_math.families.geometry_coordinates import FAMILY_NAME as GEO_FAMILY, FAMILY_TYPE as GEO_TYPE
from mira_math.families.bayes_missing_prior import FAMILY_NAME as BAYES_FAMILY, FAMILY_TYPE as BAYES_TYPE
from mira_math.families.piecewise_missing_threshold import FAMILY_NAME as PIECEWISE_FAMILY, FAMILY_TYPE as PIECEWISE_TYPE
from mira_math.families.markov_missing_transition import FAMILY_NAME as MARKOV_FAMILY, FAMILY_TYPE as MARKOV_TYPE
from mira_math.families.linear_system_missing_coeff import FAMILY_NAME as LINSYS_FAMILY, FAMILY_TYPE as LINSYS_TYPE
from mira_math.families.circuit_missing_resistance import FAMILY_NAME as CIRCUIT_FAMILY, FAMILY_TYPE as CIRCUIT_TYPE
from mira_math.families.portfolio_variance_missing_corr import FAMILY_NAME as PORTFOLIO_FAMILY, FAMILY_TYPE as PORTFOLIO_TYPE
from mira_math.families.birth_death_missing_rate import FAMILY_NAME as BIRTHDEATH_FAMILY, FAMILY_TYPE as BIRTHDEATH_TYPE
from mira_math.families.eigenvector_missing_entry import FAMILY_NAME as EIGVEC_FAMILY, FAMILY_TYPE as EIGVEC_TYPE
from mira_math.families.steady_state_missing_emission import FAMILY_NAME as SSEMIT_FAMILY, FAMILY_TYPE as SSEMIT_TYPE

FAMILIES = [
    LINEAR_FAMILY,
    CRT_FAMILY,
    POLY_FAMILY,
    REC_FAMILY,
    RANKDEF_FAMILY,
    LAPLACE_FAMILY,
    DECONV_FAMILY,
    PHASE_FAMILY,
    MATRIX_FAMILY,
    TOMO_FAMILY,
    GRAPH_FAMILY,
    MOMENT_FAMILY,
    GEO_FAMILY,
    BAYES_FAMILY,
    PIECEWISE_FAMILY,
    MARKOV_FAMILY,
    LINSYS_FAMILY,
    CIRCUIT_FAMILY,
    PORTFOLIO_FAMILY,
    BIRTHDEATH_FAMILY,
    EIGVEC_FAMILY,
    SSEMIT_FAMILY,
]

# Mapping from family name to its type ("A" = fixed hint slot, "B" = variable hint slot)
FAMILY_TYPES: dict = {
    LINEAR_FAMILY: LINEAR_TYPE,
    CRT_FAMILY: CRT_TYPE,
    POLY_FAMILY: POLY_TYPE,
    REC_FAMILY: REC_TYPE,
    RANKDEF_FAMILY: RANKDEF_TYPE,
    LAPLACE_FAMILY: LAPLACE_TYPE,
    DECONV_FAMILY: DECONV_TYPE,
    PHASE_FAMILY: PHASE_TYPE,
    MATRIX_FAMILY: MATRIX_TYPE,
    TOMO_FAMILY: TOMO_TYPE,
    GRAPH_FAMILY: GRAPH_TYPE,
    MOMENT_FAMILY: MOMENT_TYPE,
    GEO_FAMILY: GEO_TYPE,
    BAYES_FAMILY: BAYES_TYPE,
    PIECEWISE_FAMILY: PIECEWISE_TYPE,
    MARKOV_FAMILY: MARKOV_TYPE,
    LINSYS_FAMILY: LINSYS_TYPE,
    CIRCUIT_FAMILY: CIRCUIT_TYPE,
    PORTFOLIO_FAMILY: PORTFOLIO_TYPE,
    BIRTHDEATH_FAMILY: BIRTHDEATH_TYPE,
    EIGVEC_FAMILY: EIGVEC_TYPE,
    SSEMIT_FAMILY: SSEMIT_TYPE,
}

TYPE_A_FAMILIES = [f for f, t in FAMILY_TYPES.items() if t == "A"]
TYPE_B_FAMILIES = [f for f, t in FAMILY_TYPES.items() if t == "B"]
