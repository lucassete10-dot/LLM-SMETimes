"""Real tiny CPU GRU fits on synthetic cells; no public-domain accuracy claim."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from battery_smetimes.research_transfer_neural_training import train_shared_source_unit, load_shared_neural_checkpoint
from battery_smetimes.research_transfer_sequences import predict_transfer_sequence
from battery_smetimes.research_transfer_signals import RESEARCH_MODE
from scripts.prepare_same_type_forecast import save_json
from tests import test_research_transfer_sequences as fixture


class SharedNeuralTrainingTests(unittest.TestCase):
    def setUp(self):
        instance = fixture.SharedSequenceTests()
        instance.setUp()
        self.cells, self.plan, self.params = instance.cells, instance.plan, instance.parameters
        self.plan["name"] = "MIT_to_XJTU_synthetic"
        self.fold = {"name": "source_cv_fold1", "source_train": ["s1", "s2"], "source_validation": ["s3"]}
        self.plan["source_selection_folds"] = [self.fold]
        self.config = {"max_epochs": 2, "validation_every_epochs": 1, "patience_validations": 1,
            "batch_contexts": 4, "max_future_targets_per_context": 96}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fit(self, name, **kwargs):
        return train_shared_source_unit(self.cells, self.plan, [20, 50, 100], "gru", self.params, 9, "cpu", self.config,
            self.root / name, **kwargs)

    def test_source_fold_and_fixed_refit_never_read_target_values_or_labels(self):
        for key in self.plan["target_test"]:
            self.cells[key].update(common=np.full((100, 11), np.inf), soh=object(), label_cycle=object())
        bundle, report = self.fit("candidate", source_fold=self.fold)
        self.assertFalse(report["target_evaluated_during_training"])
        self.assertEqual(bundle["source_train_ids"], ["s1", "s2"])
        self.assertEqual({r["cell_id"] for r in report["source_validation_by_cell"]}, {"s3"})
        self.assertIn(report["selected_epochs"], (1, 2))
        refit, final = self.fit("final", fixed_refit_epochs=report["selected_epochs"])
        self.assertEqual(refit["source_train_ids"], self.plan["source_refit"])
        self.assertIsNone(final["source_validation_mape_percent_origin100"])
        self.assertEqual(final["source_validation_by_cell"], [])
        for name in ("candidate", "final"):
            for file in ("config.json", "training_history.json", "model.pt", "model.json", "preprocessor.joblib",
                         "resources.json", "checkpoint_manifest.json", "report.json"):
                self.assertTrue((self.root / name / file).is_file())

    def test_reloaded_checkpoint_has_exact_cpu_no_future_prediction_and_no_fit(self):
        bundle, report = self.fit("candidate", source_fold=self.fold)
        cell = copy.deepcopy(self.cells["t1"])
        expected = predict_transfer_sequence(bundle, cell, 20, [30, 80, 120], mode=RESEARCH_MODE)
        with patch("torch.optim.AdamW", side_effect=AssertionError("reload attempted fitting")):
            restored, evidence = load_shared_neural_checkpoint(self.root / "candidate", device="cpu")
        cell["common"][20:] = np.inf
        cell.pop("soh")
        cell.pop("label_cycle")
        np.testing.assert_array_equal(expected, predict_transfer_sequence(restored, cell, 20, [30, 80, 120], mode=RESEARCH_MODE))
        self.assertTrue(evidence["no_fit"])
        self.assertEqual(restored["allowed_cells"]["t1"]["physical_family"], "target-family")

    def test_changed_checkpoint_rejected_before_deserialization(self):
        self.fit("candidate", source_fold=self.fold)
        for name in ("model.pt", "model.json", "preprocessor.joblib"):
            path = self.root / "candidate" / name
            original = path.read_bytes()
            path.write_bytes(b"changed")
            try:
                with patch("joblib.load", side_effect=AssertionError("changed artifact was deserialized")):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        load_shared_neural_checkpoint(self.root / "candidate")
            finally:
                path.write_bytes(original)

    def test_target_cannot_be_early_stopping_validation_even_with_injected_fold(self):
        fold = {"name": "bad", "source_train": ["s1", "s2", "s3"], "source_validation": ["t1"]}
        self.plan["source_selection_folds"] = [fold]
        with self.assertRaisesRegex(ValueError, "source-only validation"):
            self.fit("bad", source_fold=fold)
        self.assertFalse((self.root / "bad").exists())

    def test_no_primary_validation_cell_may_be_deleted_to_improve_score(self):
        self.cells["s3"]["reference_established_source_cycle"] = 120
        with self.assertRaisesRegex(ValueError, "retain all required"):
            self.fit("candidate", source_fold=self.fold)
        failure = json.loads((self.root / "candidate/failure.json").read_text())
        self.assertFalse(failure["target_evaluated_during_training"])
        self.assertFalse((self.root / "candidate/report.json").exists())

    def test_interrupted_or_existing_attempt_never_silently_overwritten(self):
        destination = self.root / "partial"
        destination.mkdir()
        save_json(destination / "failure.json", {"interrupted": True})
        with self.assertRaises(FileExistsError):
            self.fit("partial", source_fold=self.fold)
        self.assertEqual(json.loads((destination / "failure.json").read_text()), {"interrupted": True})
        for kwargs in ({}, {"source_fold": self.fold, "fixed_refit_epochs": 1}, {"fixed_refit_epochs": 0}, {"fixed_refit_epochs": 3}):
            with self.assertRaises(ValueError):
                self.fit("invalid", **kwargs)


if __name__ == "__main__":
    unittest.main()
