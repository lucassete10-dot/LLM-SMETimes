"""Coverage and matched-future comparisons never select by prediction error."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.prepare_same_type_forecast import save_json
from scripts.report_soh_research_horizons_v5 import expected_eligibility, matched_future_report


class HorizonReportingTests(unittest.TestCase):
    def cell(self, cycles=None, reference=1):
        cycles = np.arange(1, 101) if cycles is None else np.asarray(cycles)
        return {"metadata": {"dataset": "MIT", "group": "batch1", "charge_protocol": "4C(80%)-4C",
                             "diagnostic_discharge_rate_c": 4},
                "cycle": cycles, "raw": np.ones((len(cycles), 51)),
                "reference_established_source_cycle": reference,
                "label_cycle": np.array([105, 120]), "soh": np.array([.9, .8])}

    def fixture(self, folder, omit=None, unequal=False):
        seeds, origins = [10, 11, 12, 13, 14], [20, 50, 100]
        (folder / "frozen_audit").mkdir()
        save_json(folder / "frozen_audit/splits.json", {"plans": [{"name": "MIT_fold1", "test": ["a", "b"]}]})
        save_json(folder / "frozen_audit/config.json", {"config": {
            "training_seeds": seeds, "origins_source_cycles": origins}})
        for seed in seeds:
            unit = folder / "MIT_fold1" / str(seed) / "model"
            (unit / "test").mkdir(parents=True)
            save_json(unit / "resources.json", {})
            arrays = {}
            for cell in ("a", "b"):
                for origin in origins:
                    if (seed, cell, origin) == omit:
                        continue
                    cycles = np.array([30, 60, 110, 120])
                    cycles = cycles[cycles > origin]
                    if unequal and seed == 10 and cell == "a" and origin == 20:
                        cycles[-1] = 121
                    truth = np.where(cycles <= 100, 1., .8)
                    prediction = truth + origin / 10000
                    for suffix, value in (("cycle", cycles), ("truth", truth), ("prediction", prediction)):
                        arrays[f"{cell}__{origin}__{suffix}"] = value
            np.savez_compressed(unit / "test/predictions.npz", **arrays)
        return {(origin, "all_observed_future"): {"MIT": {"a", "b"}} for origin in origins}

    def test_signal_history_and_reference_are_independent_requirements(self):
        cells = {"good": self.cell(), "few": self.cell(cycles=[1, 2]), "late_reference": self.cell(reference=65)}
        eligible, skipped = expected_eligibility(cells, 20, "all_observed_future")
        self.assertEqual(eligible, {"MIT": {"good"}})
        self.assertEqual({row["cell_id"] for row in skipped}, {"few", "late_reference"})

    def test_short_horizon_requires_real_diagnostic_not_interpolation(self):
        cell = self.cell()
        cell["label_cycle"] = np.array([120, 130])
        eligible, skipped = expected_eligibility({"a": cell}, 100, "within_10")
        self.assertEqual(eligible["MIT"], set())
        self.assertIn("no comparable diagnostic", skipped[0]["reason"])
        self.assertEqual(expected_eligibility({"a": cell}, 100, "within_50")[0]["MIT"], {"a"})

    def test_missing_future_labels_remain_coverage_failures(self):
        cell = self.cell()
        cell["soh"][:] = np.nan
        eligible, skipped = expected_eligibility({"a": cell}, 100, "all_observed_future")
        self.assertEqual(eligible["MIT"], set())
        self.assertEqual(skipped[0]["reason"], "insufficient valid future targets")

    def test_invalid_visible_signals_are_not_mislabeled_as_too_little_history(self):
        cell = self.cell()
        cell["raw"][0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "invalid signal history"):
            expected_eligibility({"a": cell}, 100, "all_observed_future")

    def test_matched_comparison_uses_same_cells_and_future_grid(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            grid = self.fixture(folder)
            rows = matched_future_report(folder, grid)
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertTrue(row["complete_coverage"])
                self.assertEqual(row["evaluation_cells"], 2)
                self.assertEqual(row["completed_cell_seed_pairs"], 10)
                score = row["metrics"]["mape_percent"]["seed_macro_mean_and_sd"]
                self.assertAlmostEqual(score["mean"], row["origin_source_cycle"] / 80)
                self.assertAlmostEqual(score["sd"], 0)

    def test_missing_seed_cell_origin_never_receives_complete_score(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            grid = self.fixture(folder, omit=(14, "b", 20))
            row = next(r for r in matched_future_report(folder, grid) if r["origin_source_cycle"] == 20)
            self.assertFalse(row["complete_coverage"])
            self.assertNotIn("metrics", row)
            self.assertEqual(row["missing"], [(14, "b")])

    def test_unequal_future_grids_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            grid = self.fixture(folder, unequal=True)
            with self.assertRaisesRegex(ValueError, "unequal future target grids"):
                matched_future_report(folder, grid)

    def test_no_common_cells_does_not_mean_perfect_prediction(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            grid = self.fixture(folder)
            grid[(20, "all_observed_future")]["MIT"] = set()
            for row in matched_future_report(folder, grid):
                self.assertFalse(row["complete_coverage"])
                self.assertEqual(row["evaluation_cells"], 0)
                self.assertNotIn("metrics", row)


if __name__ == "__main__":
    unittest.main()
