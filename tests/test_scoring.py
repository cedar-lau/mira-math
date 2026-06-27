import json
import os
import unittest

from mira_math.scoring import score_transcript

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class ScoringTests(unittest.TestCase):
    def test_sample_transcript(self):
        with open(os.path.join(REPO_ROOT, "examples", "sample_instance.json"), "r", encoding="utf-8") as f:
            inst = json.load(f)
        with open(os.path.join(REPO_ROOT, "examples", "sample_transcript.json"), "r", encoding="utf-8") as f:
            transcript = json.load(f)
        metrics = score_transcript(inst, transcript)
        self.assertEqual(metrics["acc_final"], 1)
        self.assertEqual(metrics["hints_used"], 1)


if __name__ == "__main__":
    unittest.main()
