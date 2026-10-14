import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.report_soh_research_primary_v5 import report_run, render_primary_table


class PrimaryReportingTests(unittest.TestCase):
    def fixture(self, folder, omit=None):
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        seeds = [10, 11, 12, 13, 14]
        save(folder / "config.json", {"smoke": False})
        plans = [{"name": f"MIT_fold{i + 1}", "test": [f"cell{j}" for j in range(2 * i + 1, min(2 * i + 3, 10))]}
                 for i in range(5)]
        save(folder / "frozen_audit/splits.json", {"plans": plans})
        save(folder / "frozen_audit/config.json", {"config": {"training_seeds": seeds}})
        for plan in plans:
            for seed in seeds:
                rows = [{"cell_id": cell, "origin": 100,
                         "metrics": {"mape_percent": int(cell[4:]) + seed - 10,
                                     "rmse_soh": .01, "mae_soh": .005, "r2": None}}
                        for cell in plan["test"] if (seed, cell) != omit]
                save(folder / plan["name"] / str(seed) / "test_model/test/metrics.json", {"by_cell": rows})
                save(folder / plan["name"] / str(seed) / "test_model/resources.json", {})

    def test_primary_is_cell_macro_not_unweighted_fold_mean(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            report = report_run(folder)[0]
            metric = report["metrics"]["mape_percent"]
            self.assertTrue(report["complete_model_dataset_primary"])
            self.assertEqual(report["expected_cells"], 9)
            self.assertEqual(metric["seed_macro_mean_and_sd"]["mean"], 7)
            self.assertAlmostEqual(metric["seed_macro_mean_and_sd"]["sd"], np.sqrt(2.5))
            self.assertAlmostEqual(metric["fold_macro_descriptive_mean_and_sd_not_independent_CI"]["mean"], 7.4)
            self.assertIsNone(report["metrics"]["r2"]["seed_macro_mean_and_sd"]["mean"])

    def test_partial_cell_seed_coverage_has_no_primary_number(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder, omit=(14, "cell9"))
            report = report_run(folder)[0]
            self.assertFalse(report["complete_model_dataset_primary"])
            self.assertNotIn("metrics", report)
            self.assertEqual(report["missing_cell_seed_pairs"], [(14, "cell9")])

    def test_smoke_is_never_in_primary_table(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "config.json").write_text('{"smoke": true}', encoding="utf-8")
            self.assertEqual(report_run(folder), [])

    def test_independent_eligibility_preserves_full_denominator(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            report = report_run(folder, eligible_cells={"MIT": {"cell1", "cell2"}})[0]
            self.assertTrue(report["complete_model_dataset_primary"])
            self.assertEqual(report["full_cohort_cells"], 9)
            self.assertEqual(report["expected_cells"], 2)
            self.assertAlmostEqual(report["metrics"]["mape_percent"]["seed_macro_mean_and_sd"]["mean"], 3.5)

    def test_nonfinite_error_is_not_silently_averaged_away(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            path = folder / "MIT_fold1/10/test_model/test/metrics.json"
            value = json.loads(path.read_text())
            value["by_cell"][0]["metrics"]["mape_percent"] = float("inf")
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid regression metric"):
                report_run(folder)

    def test_duplicate_frozen_test_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            path = folder / "frozen_audit/splits.json"
            value = json.loads(path.read_text())
            value["plans"][1]["test"].append("cell1")
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "more than one frozen fold"):
                report_run(folder)

    def test_ineligible_battery_still_cannot_use_wrong_fold(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            path = folder / "MIT_fold1/10/test_model/test/metrics.json"
            value = json.loads(path.read_text())
            value["by_cell"][1]["cell_id"] = "cell9"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test membership"):
                report_run(folder, eligible_cells={"MIT": {"cell1"}})

    def test_unfinalized_outputs_are_not_read(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            (folder / "MIT_fold5/14/test_model/resources.json").unlink()
            report = report_run(folder)[0]
            self.assertFalse(report["complete_model_dataset_primary"])
            self.assertNotIn("metrics", report)
            self.assertEqual(report["missing_cell_seed_pairs"], [(14, "cell9")])

    def test_table_displays_four_metrics_without_mutating_source_numbers(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            rows = report_run(folder)
            original = copy.deepcopy(rows)
            table = render_primary_table(rows)
            self.assertIn("MAE（pp）", table)
            self.assertIn("RMSE（pp）", table)
            self.assertIn("MAPE（%）", table)
            self.assertIn("R²", table)
            self.assertIn("| 9/9 | 0.500 ± 0.000 | 1.000 ± 0.000 | 7.000 ± 1.581 | 不适用 | 0/9 |", table)
            self.assertIn("cell9 | 11.000 ± 1.581", table)
            self.assertEqual(rows, original)

    def test_partial_model_has_neither_primary_numerals_nor_worst_case_in_table(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder, omit=(14, "cell9"))
            table = render_primary_table(report_run(folder))
            self.assertNotIn("| MIT | test_model |", table)

    def test_negative_r2_is_preserved_and_defined_cell_count_is_shown(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            for path in folder.glob("*/*/*/test/metrics.json"):
                value = json.loads(path.read_text())
                for row in value["by_cell"]:
                    row["metrics"]["r2"] = -.75
                path.write_text(json.dumps(value), encoding="utf-8")
            table = render_primary_table(report_run(folder))
            self.assertIn("| -0.750 ± 0.000 | 9/9 |", table)

    def test_rendering_rejects_nonfinite_summary_instead_of_printing_it(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            rows = report_run(folder)
            rows[0]["metrics"]["mae_soh"]["seed_macro_mean_and_sd"]["mean"] = float("nan")
            with self.assertRaisesRegex(ValueError, "nonfinite statistics"):
                render_primary_table(rows)


if __name__ == "__main__":
    unittest.main()
