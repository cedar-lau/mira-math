import json
import os
import unittest

from mira_math.validate import validate_instance

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class ValidatorTests(unittest.TestCase):
    def test_sample_instance(self):
        with open(os.path.join(REPO_ROOT, "examples", "sample_instance.json"), "r", encoding="utf-8") as f:
            inst = json.load(f)
        errs = validate_instance(inst)
        self.assertEqual(errs, [])


if __name__ == "__main__":
    unittest.main()
