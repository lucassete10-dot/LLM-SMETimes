import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.resume_soh_research_v5 import ROOT, archive_partial, scores_complete


class ResearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="recovery-unit-test-", dir=ROOT / "experiments/soh_research_v5")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.model = self.run / "MIT_fold1/123/test_model"
        self.test = self.model / "test"
        self.test.mkdir(parents=True)
        self.arrays = {"cell__100__cycle": np.array([110, 120, 130]),
                       "cell__100__truth": np.array([.9, .89, .88]),
                       "cell__100__prediction": np.array([.91, .90, .89])}
        (self.test / "metrics.json").write_text(json.dumps({"by_cell": [{"cell_id": "cell", "origin": 100, "future_targets": 3}]}), encoding="utf-8")
        np.savez(self.test / "predictions.npz", **self.arrays)

    def test_complete_saved_scores_are_reusable(self):
        self.assertTrue(scores_complete(self.model, "test"))

    def test_partial_archive_is_not_complete(self):
        (self.test / "predictions.npz").write_bytes(b"interrupted")
        self.assertFalse(scores_complete(self.model, "test"))

    def test_wrong_prediction_count_is_not_complete(self):
        self.arrays["cell__100__prediction"] = np.ones(2)
        np.savez(self.test / "predictions.npz", **self.arrays)
        self.assertFalse(scores_complete(self.model, "test"))

    def test_nonfinite_prediction_is_not_complete(self):
        self.arrays["cell__100__prediction"][0] = np.inf
        np.savez(self.test / "predictions.npz", **self.arrays)
        self.assertFalse(scores_complete(self.model, "test"))

    def test_move_preserves_attempt_under_exact_run(self):
        target = Path(archive_partial(self.test, self.run, "unit-session"))
        self.assertTrue(target.is_relative_to(self.run))
        self.assertTrue((target / "metrics.json").is_file())
        self.assertTrue((target / "predictions.npz").is_file())
        self.assertFalse(self.test.exists())

    def test_refuses_run_root_as_a_move_target(self):
        with self.assertRaises(ValueError):
            archive_partial(self.run, self.run, "unit-session")
        self.assertTrue(self.run.exists())


if __name__ == "__main__":
    unittest.main()
