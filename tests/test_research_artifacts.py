"""Prediction files must substantiate their claimed numbers and coverage."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from battery_smetimes.research_baselines import regression_metrics
from scripts.audit_soh_research_artifacts_v5 import audit_scores, audit_selection, baseline_candidates, collect_replay_failures, copy_source_snapshot
from scripts.prepare_same_type_forecast import save_json


class ArtifactAuditTests(unittest.TestCase):
    def fixture(self, folder):
        cycle, truth, prediction = np.array([101, 110, 130]), np.array([.9, .85, .8]), np.array([.91, .86, .81])
        row = {"cell_id": "held_out", "origin": 100, "physical_family": "family", "future_targets": 3,
               "metrics": regression_metrics(truth, prediction),
               "within_horizon": {str(h): regression_metrics(truth[cycle <= 100 + h], prediction[cycle <= 100 + h]) for h in (10, 50, 100)},
               "late_half": regression_metrics(truth[cycle >= 115.5], prediction[cycle >= 115.5])}
        report = {"by_cell": [row], "skipped": [{"cell_id": "no_history", "origin": 100, "reason": "insufficient signal history"}],
                  "summary": {"100": {"cells": 1, "cell_macro_mape_percent": row["metrics"]["mape_percent"], "fraction_cells_mape_le_6": 1}}}
        arrays = {f"held_out__100__{name}": value for name, value in (("cycle", cycle), ("truth", truth), ("prediction", prediction))}
        self.save(folder, report, arrays)
        return report, arrays

    @staticmethod
    def save(folder, report, arrays):
        save_json(folder / "metrics.json", report)
        np.savez_compressed(folder / "predictions.npz", **arrays)

    def audit(self, folder, cells=None):
        return audit_scores(folder, ["held_out", "no_history"], [100], "family", cells)

    def test_valid_arrays_numbers_and_skip_coverage_pass(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            report, _ = self.fixture(folder)
            self.assertEqual(self.audit(folder), report)

    def test_removed_hard_cell_without_reason_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            report, arrays = self.fixture(folder)
            report["skipped"] = []
            self.save(folder, report, arrays)
            with self.assertRaisesRegex(ValueError, "disappeared"):
                self.audit(folder)

    def test_invented_good_error_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            report, arrays = self.fixture(folder)
            report["by_cell"][0]["metrics"]["mape_percent"] = .01
            self.save(folder, report, arrays)
            with self.assertRaisesRegex(ValueError, "differs from prediction arrays"):
                self.audit(folder)

    def test_duplicate_prediction_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            report, arrays = self.fixture(folder)
            report["by_cell"].append(copy.deepcopy(report["by_cell"][0]))
            self.save(folder, report, arrays)
            with self.assertRaisesRegex(ValueError, "duplicate prediction"):
                self.audit(folder)

    def test_future_grid_cannot_include_visible_cycle(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            report, arrays = self.fixture(folder)
            arrays["held_out__100__cycle"][0] = 100
            self.save(folder, report, arrays)
            with self.assertRaisesRegex(ValueError, "invalid future target grid"):
                self.audit(folder)

    def test_wrong_ground_truth_is_rejected_even_if_metrics_match(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            _, arrays = self.fixture(folder)
            cells = {"held_out": {"label_cycle": arrays["held_out__100__cycle"], "soh": arrays["held_out__100__truth"] + .1}}
            with self.assertRaisesRegex(ValueError, "audited prepared targets"):
                self.audit(folder, cells)

    def test_unreported_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            report, arrays = self.fixture(folder)
            arrays["secret__100__prediction"] = np.array([1, 1])
            self.save(folder, report, arrays)
            with self.assertRaisesRegex(ValueError, "unreported or missing"):
                self.audit(folder)

    def test_corrupt_short_horizon_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            report, arrays = self.fixture(folder)
            report["by_cell"][0]["within_horizon"]["10"]["mae_soh"] = .001
            self.save(folder, report, arrays)
            with self.assertRaisesRegex(ValueError, "within_10/mae_soh"):
                self.audit(folder)

    def test_saved_selection_uses_only_minimum_validation_score(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            candidates = [{"alpha": 1}, {"alpha": 10}]
            scores = [{"index": i, "parameters": candidate, "validation_mape_percent": value}
                      for i, (candidate, value) in enumerate(zip(candidates, [3., 2.]))]
            reports = [{"summary": {"100": {"cell_macro_mape_percent": value}}} for value in [3., 2.]]
            save_json(folder / "validation_scores.json", scores)
            save_json(folder / "selected.json", scores[1])
            self.assertEqual(audit_selection(folder, candidates, reports), scores[1])
            save_json(folder / "selected.json", scores[0])
            with self.assertRaisesRegex(ValueError, "minimum validation-only"):
                audit_selection(folder, candidates, reports)

    def test_absolute_source_path_cannot_replace_snapshot_destination(self):
        with tempfile.TemporaryDirectory() as name:
            workspace = Path(name).resolve()
            source = workspace / "scripts/source.py"
            source.parent.mkdir()
            source.write_text("original immutable source", encoding="utf-8")
            output = workspace / "audit"
            target = copy_source_snapshot(source, output, workspace)
            self.assertEqual(target, output / "source_snapshot/scripts/source.py")
            self.assertEqual(target.read_text(), "original immutable source")
            self.assertEqual(source.read_text(), "original immutable source")

    def test_snapshot_never_overwrites_an_existing_copy(self):
        with tempfile.TemporaryDirectory() as name:
            workspace = Path(name).resolve()
            source = workspace / "source.py"
            source.write_text("original", encoding="utf-8")
            output = workspace / "audit"
            copy_source_snapshot(source, output, workspace)
            with self.assertRaisesRegex(ValueError, "already exists"):
                copy_source_snapshot(source, output, workspace)

    def test_snapshot_source_outside_named_workspace_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            source = root / "outside.py"
            source.write_text("not a project source", encoding="utf-8")
            with self.assertRaises(ValueError):
                copy_source_snapshot(source, root / "audit", root / "workspace")

    def test_historical_search_config_is_not_replaced_by_current_defaults(self):
        old = {"config": {"initial_candidates": {"ridge": [{"alpha": 100}]}}}
        self.assertEqual(baseline_candidates(old), {"ridge": [{"alpha": 100}]})
        new = {**old, "model_candidates": {"gpr": [{"alpha": .1}]}}
        self.assertEqual(baseline_candidates(new), {"gpr": [{"alpha": .1}]})

    def test_failed_portability_check_is_preserved_in_inventory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            replay = root / "attempt"
            replay.mkdir()
            model = root / "model"
            save_json(replay / "config.json", {"model_folder": str(model), "device": "cpu", "original_training_device": "cuda"})
            save_json(replay / "failure.json", {"traceback": "failed strict replay", "rows": [{"passed": False, "same_device_no_future_invariance": True}]})
            records = collect_replay_failures(root)[str(model.resolve())]
            self.assertEqual(records[0]["device"], "cpu")
            self.assertEqual(records[0]["original_training_device"], "cuda")
            self.assertTrue(records[0]["failed_rows"][0]["same_device_no_future_invariance"])


if __name__ == "__main__":
    unittest.main()
