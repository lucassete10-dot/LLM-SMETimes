"""Synthetic metric-grid reporting tests, not public transfer accuracy."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from battery_smetimes.research_transfer_reporting import (
    aggregate_cell_errors, build_transfer_tables, validate_comparison_contracts, MODELS,
)
from battery_smetimes.research_transfer_baselines import evaluate_transfer, transfer_prefix
from scripts.report_soh_research_shared_v5 import read_completed_run, read_audited_units, write_tables
from scripts.prepare_same_type_forecast import save_json
from battery_smetimes.research_transfer_vi120 import stream_sha
from tests import test_research_transfer_execution as fixture_module


class TransferReportingTests(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.TransferFitTests()
        fixture.setUp()
        self.cells = fixture.cells
        plan = {**fixture.plan, "name": "MIT_to_NASA_synthetic", "evidence_level": "synthetic only",
            "not_same_type_deployment": True}
        reverse = {**plan, "name": "NASA_to_MIT_synthetic", "source_dataset": "NASA", "target_dataset": "MIT",
            "source_physical_family": "target-family", "target_physical_family": "source-family",
            "source_refit": plan["target_test"], "target_test": plan["source_refit"]}
        self.plans = {key: [plan, reverse] for key in MODELS}
        shared = {"smoke": False, "prepare_only": False, "target_calibration": [], "not_same_type_deployment": True,
            "global_summary_writer": False, "prepared": "synthetic", "prepared_config_sha256": "toy",
            "prepared_manifest_sha256": "toy", "prepared_splits_sha256": "toy", "main_run_bindings": [],
            "origins": [20, 50, 100], "seeds": [10, 11, 12, 13, 14], "input_schema": "synthetic", "feature_names": ["toy"]}
        self.configs = {key: {**shared, "models": models} for key, models in MODELS.items()}
        self.cells["t2"]["reference_established_source_cycle"] = 65
        self.units = {}
        for protocol, config in self.configs.items():
            for plan in self.plans[protocol]:
                for seed in shared["seeds"]:
                    # Simulated saved predictions. No claim of trained models.
                    def predictions(bundle, cell, origin, future, **kwargs):
                        transfer_prefix(cell, origin)
                        truth = np.interp(future, cell["label_cycle"], cell["soh"])
                        return truth*(1+(.50 if cell["cell_id"] == "t2" else .01*(seed-9)))
                    report, arrays = evaluate_transfer({"origins": shared["origins"]}, self.cells, plan["target_test"], predictor=predictions)
                    for model in config["models"]:
                        self.units[(protocol, plan["name"], seed, model)] = {"metrics": copy.deepcopy(report),
                            "arrays": {key: value.copy() for key, value in arrays.items()}}

    def test_full_tables_preserve_hard_cell_coverage_and_paired_future_controls(self):
        report = build_transfer_tables(self.configs, self.plans, self.cells, self.units)
        self.assertEqual(len(report["primary"]), 180)
        self.assertEqual(len(report["matched_prefix_and_future"]), 36)
        self.assertEqual(len(report["coverage"]), 30)
        rows = [row for row in report["primary"] if row["direction"] == "MIT_to_NASA_synthetic" and row["model"] == "ridge"
            and row["horizon"] == "all_observed_future"]
        self.assertEqual([row["evaluation_cells"] for row in rows], [1, 1, 2])
        self.assertEqual(rows[-1]["worst_cell"]["cell_id"], "t2")
        self.assertAlmostEqual(rows[-1]["metrics"]["mape_percent"]["seed_macro_mean_and_sd"]["mean"], 26.5)
        self.assertEqual(rows[-1]["fraction_cells_seed_averaged_mape_le_6"], .5)
        matched = [row for row in report["matched_prefix_and_future"] if row["direction"] == "MIT_to_NASA_synthetic"]
        self.assertTrue(all(row["evaluation_cells"] == 1 and row["coverage_fraction"] == .5 for row in matched))
        self.assertTrue(all(row["not_same_type_deployment"] and row["target_calibration"] == [] for row in report["primary"]))
        with tempfile.TemporaryDirectory() as name:
            counts = write_tables(Path(name), report)
            self.assertEqual(counts["summary_rows"], 216)
            self.assertTrue((Path(name) / "by_cell.csv").is_file())

    def test_macro_equal_cell_weight_seed_sd_and_undefined_r2(self):
        seeds, ids = [10, 11, 12, 13, 14], ["a", "b"]
        scores = {(seed, key): {"rmse_soh": .01, "mae_soh": .005, "r2": None,
            "mape_percent": (1 if key == "a" else 100)+seed-10} for seed in seeds for key in ids}
        report = aggregate_cell_errors(scores, ids, seeds)
        self.assertEqual(report["metrics"]["mape_percent"]["seed_macro_mean_and_sd"]["mean"], 52.5)
        self.assertAlmostEqual(report["metrics"]["mape_percent"]["seed_macro_mean_and_sd"]["sd"], np.sqrt(2.5))
        self.assertIsNone(report["metrics"]["r2"]["seed_macro_mean_and_sd"]["mean"])
        self.assertEqual(report["metrics"]["r2"]["undefined_cell_seed_pairs"], 10)
        self.assertNotIn("fold_macro_values", report["metrics"]["mape_percent"])
        scores[(10, "b")]["mape_percent"] = np.inf
        with self.assertRaisesRegex(ValueError, "invalid regression"):
            aggregate_cell_errors(scores, ids, seeds)

    def test_smoke_calibration_seed_input_or_role_changes_cannot_be_pooled(self):
        for field, value in (("smoke", True), ("prepare_only", True), ("target_calibration", ["t1"]),
                             ("not_same_type_deployment", False), ("seeds", [10]), ("input_schema", "other"),
                             ("feature_names", ["other"]), ("prepared_manifest_sha256", "other")):
            configs = copy.deepcopy(self.configs)
            configs["neural"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_comparison_contracts(configs, self.plans)
        plans = copy.deepcopy(self.plans)
        plans["neural"][0]["target_test"].append("s1")
        with self.assertRaises(ValueError):
            validate_comparison_contracts(self.configs, plans)

    def test_missing_grid_duplicate_cell_or_error_based_skip_rejected(self):
        units = self.units.copy()
        key = next(iter(units))
        units.pop(key)
        with self.assertRaisesRegex(ValueError, "entire exact"):
            build_transfer_tables(self.configs, self.plans, self.cells, units)
        for mutation in ("duplicate", "skip", "missing_metric"):
            units = copy.deepcopy(self.units)
            report = units[key]["metrics"]
            row = report["by_cell"][0]
            if mutation == "duplicate":
                report["by_cell"].append(copy.deepcopy(row))
            elif mutation == "skip":
                report["by_cell"].remove(row)
                report["skipped"].append({"cell_id": row["cell_id"], "origin": row["origin"], "reason": "large prediction error"})
            else:
                row["within_horizon"]["10"] = None
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_transfer_tables(self.configs, self.plans, self.cells, units)

    def test_matched_prefix_comparison_rejects_different_future_truth_or_grid(self):
        key = next(iter(self.units))
        for name in ("cycle", "truth"):
            units = copy.deepcopy(self.units)
            array = units[key]["arrays"]["t1__20__"+name]
            array[-1] += 1 if name == "cycle" else .001
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unequal future"):
                build_transfer_tables(self.configs, self.plans, self.cells, units)

    def test_no_eligible_targets_has_no_zero_or_fabricated_macro(self):
        for key in ("t1", "t2"):
            self.cells[key]["reference_established_source_cycle"] = 120
        for protocol, config in self.configs.items():
            plan = self.plans[protocol][0]
            for seed in config["seeds"]:
                report, arrays = evaluate_transfer({"origins": config["origins"]}, self.cells, plan["target_test"],
                    predictor=lambda bundle, cell, origin, future, **kwargs: (_ for _ in ()).throw(ValueError("capacity reference is not established")))
                # Use the exact independently rederived reason.
                for row in report["skipped"]:
                    try:
                        transfer_prefix(self.cells[row["cell_id"]], row["origin"])
                    except ValueError as error:
                        row["reason"] = str(error)
                for model in config["models"]:
                    self.units[(protocol, plan["name"], seed, model)] = {"metrics": report, "arrays": arrays}
        report = build_transfer_tables(self.configs, self.plans, self.cells, self.units)
        rows = [row for row in report["primary"] if row["direction"] == "MIT_to_NASA_synthetic"]
        self.assertTrue(all(row["evaluation_cells"] == 0 and "metrics" not in row for row in rows))

    def test_completed_run_guard_rejects_prepare_before_any_weight_load(self):
        with tempfile.TemporaryDirectory() as name:
            workspace = Path(name)
            run = workspace / "experiments/soh_research_v5/transfer_baseline_runs/toy"
            run.mkdir(parents=True)
            save_json(run / "progress.json", {"complete": False, "model_training_started": False})
            with patch("scripts.report_soh_research_shared_v5.verify_source_snapshot", return_value={"prepare_only": True}), \
                 patch("joblib.load", side_effect=AssertionError("report deserialized")):
                with self.assertRaisesRegex(ValueError, "complete real fitting"):
                    read_completed_run(run, "baseline", workspace)
            with self.assertRaisesRegex(ValueError, "matching trusted-local"):
                read_completed_run(run, "neural", workspace)

    def test_changed_audited_predictions_rejected_before_reading_npz(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            plan = self.plans["baseline"][0]
            config = {"seeds": [10], "models": ["ridge"]}
            unit = run / plan["name"] / "10/ridge"
            target = unit / "target_test"
            target.mkdir(parents=True)
            save_json(target / "metrics.json", {})
            np.savez_compressed(target / "predictions.npz", x=np.array([1.]))
            save_json(unit / "static_unit_audit.json", {"artifact_sha256": {
                str(Path("target_test") / "metrics.json"): stream_sha(target / "metrics.json"),
                str(Path("target_test") / "predictions.npz"): "wrong"}})
            with patch("numpy.load", side_effect=AssertionError("changed predictions read")):
                with self.assertRaisesRegex(ValueError, "evidence changed"):
                    read_audited_units(run, "baseline", config, [plan],
                        {(plan["name"], 10, "ridge"): stream_sha(unit / "static_unit_audit.json")})

    def test_changed_unit_audit_cannot_replace_complete_run_reference(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            plan = self.plans["baseline"][0]
            unit = run / plan["name"] / "10/ridge"
            unit.mkdir(parents=True)
            save_json(unit / "static_unit_audit.json", {"changed": True})
            with patch("numpy.load", side_effect=AssertionError("changed audit used")):
                with self.assertRaisesRegex(ValueError, "after complete-run"):
                    read_audited_units(run, "baseline", {"seeds": [10], "models": ["ridge"]}, [plan],
                        {(plan["name"], 10, "ridge"): "original"})


if __name__ == "__main__":
    unittest.main()
