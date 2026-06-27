import unittest
import random

from mira_math.generate import generate_instances
from mira_math.validate import validate_instance
from mira_math.families.linear import FAMILY_NAME as LINEAR_FAMILY
from mira_math.families.crt import FAMILY_NAME as CRT_FAMILY
from mira_math.families.poly_interp import FAMILY_NAME as POLY_FAMILY
from mira_math.families.recurrence import FAMILY_NAME as REC_FAMILY
from mira_math.families.rankdef_linear_shared import FAMILY_NAME as RANKDEF_FAMILY
from mira_math.families.laplace_grid import FAMILY_NAME as LAPLACE_FAMILY
from mira_math.families.deconvolution import FAMILY_NAME as DECONV_FAMILY
from mira_math.families.phase_retrieval import FAMILY_NAME as PHASE_FAMILY
from mira_math.families.matrix_completion import FAMILY_NAME as MATRIX_FAMILY
from mira_math.families.tomography import FAMILY_NAME as TOMO_FAMILY
from mira_math.families.graph_path_sums import FAMILY_NAME as GRAPH_FAMILY
from mira_math.families.moment_problem import FAMILY_NAME as MOMENT_FAMILY
from mira_math.families.geometry_coordinates import FAMILY_NAME as GEO_FAMILY
from mira_math.families.bayes_missing_prior import FAMILY_NAME as BAYES_FAMILY
from mira_math.families.piecewise_missing_threshold import FAMILY_NAME as PIECEWISE_FAMILY
from mira_math.families.markov_missing_transition import FAMILY_NAME as MARKOV_FAMILY


class GeneratorTests(unittest.TestCase):
    def test_generate_and_validate(self):
        families = [
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
        ]
        for fam in families:
            inst = generate_instances(fam, n=1, seed=0, difficulty=1)[0]
            errs = validate_instance(inst)
            self.assertEqual(errs, [], msg=f"{fam} errors: {errs}")


if __name__ == "__main__":
    unittest.main()
