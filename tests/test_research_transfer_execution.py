import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from battery_smetimes.research_phase_gate import (
    REQUIRED_MAIN_MODELS, assert_main_complete, assert_main_numerical_audits, select_source_candidate,
)
from battery_smetimes.research_transfer_baselines import (
    evaluate_transfer, fit_transfer, predict_transfer, transfer_prefix,
)
from battery_smetimes.research_transfer_signals import FEATURE_NAMES, RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import SCHEMA as VI_SCHEMA, VI_FEATURE_NAMES
from scripts.prepare_same_type_forecast import save_json, sha


class TransferFitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.cells = {key: {"cell_id": key, "cycle": np.arange(1, 101),
            "common": rng.normal(size=(100, len(FEATURE_NAMES))),
            "metadata": {"dataset": "MIT" if key.startswith("s") else "NASA"},
            "physical_family": "source-family" if key.startswith("s") else "target-family",
            "reference_established_source_cycle": 1, "label_cycle": np.arange(1, 161),
            "soh": np.linspace(1, .75, 160)} for key in ("s1", "s2", "s3", "t1", "t2")}
        self.plan = {"mode": RESEARCH_MODE, "source_dataset": "MIT", "target_dataset": "NASA",
            "source_physical_family": "source-family", "target_physical_family": "target-family",
            "source_refit": ["s1", "s2", "s3"], "target_test": ["t1", "t2"], "target_calibration": []}

    def fit(self, model="ridge"):
        parameters = {"alpha": 10.} if model == "ridge" else {"min_samples_leaf": 2}
        return fit_transfer(self.cells, ["s1", "s2"], self.plan, [20, 50, 100], model, parameters, 9, smoke=True)

    def test_target_data_cannot_fit_source_preprocessors_or_regression(self):
        before = self.fit()
        for key in ("t1", "t2"):
            self.cells[key]["common"][:] = 1e6
            self.cells[key]["soh"] = object()  # would fail if target labels were read
        after = self.fit()
        np.testing.assert_array_equal(before["imputer"].statistics_, after["imputer"].statistics_)
        np.testing.assert_array_equal(before["scaler"].mean_, after["scaler"].mean_)
        np.testing.assert_array_equal(before["model"].coef_, after["model"].coef_)
        self.assertEqual(after["source_only_preprocessor_fit_ids"], ["s1", "s2"])

    def test_target_or_duplicate_ids_cannot_enter_fitting(self):
        for ids in (["s1", "t1"], ["s1", "s1"], []):
            with self.assertRaisesRegex(ValueError, "cannot enter"):
                fit_transfer(self.cells, ids, self.plan, [20, 50, 100], "ridge", {"alpha": 10.}, 9)

    def test_prediction_ignores_labels_and_future_shared_rows(self):
        bundle = self.fit()
        cell = self.cells["t1"]
        before = predict_transfer(bundle, cell, 20, [30, 80, 120], mode=RESEARCH_MODE)
        cell["common"][20:] = np.inf
        cell.pop("soh")
        cell.pop("label_cycle")
        np.testing.assert_array_equal(before, predict_transfer(bundle, cell, 20, [30, 80, 120], mode=RESEARCH_MODE))

    def test_production_or_renamed_family_cannot_use_research_route(self):
        bundle = self.fit()
        with self.assertRaisesRegex(ValueError, "research-only"):
            predict_transfer(bundle, self.cells["t1"], 100, [110], mode="deployment")
        self.cells["t1"]["physical_family"] = "source-family"
        with self.assertRaisesRegex(ValueError, "unchanged frozen"):
            predict_transfer(bundle, self.cells["t1"], 100, [110], mode=RESEARCH_MODE)

    def test_reference_short_history_and_invalid_signal_are_distinct(self):
        cell = self.cells["s1"]
        cell["reference_established_source_cycle"] = 65
        with self.assertRaisesRegex(ValueError, "not established"):
            transfer_prefix(cell, 50)
        cell["reference_established_source_cycle"] = 1
        with self.assertRaisesRegex(ValueError, "insufficient"):
            transfer_prefix(cell, 2)
        cell["common"][0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "invalid shared visible"):
            transfer_prefix(cell, 50)

    def test_whole_curve_and_single_future_query_are_consistent(self):
        for model in ("ridge", "extra_trees"):
            bundle = self.fit(model)
            whole = predict_transfer(bundle, self.cells["t1"], 100, [110, 120], mode=RESEARCH_MODE)
            single = predict_transfer(bundle, self.cells["t1"], 100, [110], mode=RESEARCH_MODE)
            np.testing.assert_allclose(whole[:1], single, rtol=1e-10, atol=1e-10)

    def test_target_errors_and_coverage_are_retained_not_used_for_filtering(self):
        bundle = self.fit()
        self.cells["t2"]["soh"][:] = .01
        report, arrays = evaluate_transfer(bundle, self.cells, ["t1", "t2"])
        self.assertEqual(report["summary"]["100"]["cells"], 2)
        worst = next(r for r in report["by_cell"] if r["cell_id"] == "t2" and r["origin"] == 100)
        self.assertGreater(worst["metrics"]["mape_percent"], 6)
        self.assertIn("t2__100__prediction", arrays)

    def test_equal_width_vi_vit_cohorts_cannot_mix_in_fit(self):
        self.cells["t1"].update(input_schema=VI_SCHEMA, feature_names=VI_FEATURE_NAMES)
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            self.fit()

    def test_schema_cannot_change_between_fit_and_query(self):
        bundle = self.fit()
        self.cells["t1"].update(input_schema=VI_SCHEMA, feature_names=VI_FEATURE_NAMES)
        with self.assertRaisesRegex(ValueError, "query protocol"):
            predict_transfer(bundle, self.cells["t1"], 100, [110], mode=RESEARCH_MODE)

    def test_vi_bundle_predicts_without_capacity_or_temperature_inputs(self):
        for cell in self.cells.values():
            cell.update(input_schema=VI_SCHEMA, feature_names=VI_FEATURE_NAMES)
        bundle = self.fit()
        self.assertEqual(bundle["input_schema"], VI_SCHEMA)
        prediction = predict_transfer(bundle, self.cells["t1"], 100, [110], mode=RESEARCH_MODE)
        self.assertTrue(np.isfinite(prediction).all())
        self.cells["t2"]["soh"][:] = np.nan
        report, _ = evaluate_transfer(bundle, self.cells, ["t1", "t2"])
        self.assertEqual(report["summary"]["100"]["cells"], 1)
        self.assertEqual({r["cell_id"] for r in report["skipped"]}, {"t2"})


class SourceSelectionAndPhaseTests(unittest.TestCase):
    def candidates(self):
        return [{"index": index, "parameters": {"alpha": index + 1},
            "by_cell": [{"cell_id": cell, "origin": 100, "metrics": {"mape_percent": value}}
                        for cell, value in zip(("a", "b"), values)]}
            for index, values in enumerate(([2, 4], [3, 1], [10, 10]))]

    def phase_fixture(self, run):
        (run / "source_snapshot").mkdir()
        code = run / "source_snapshot/fixture_code.py"
        code.write_text("# fixture, not a real trained model\n", encoding="utf-8")
        folds, seeds = [f"MIT_fold{j}" for j in range(1, 6)], list(range(10, 15))
        models = sorted(REQUIRED_MAIN_MODELS)
        (run / "frozen_audit").mkdir()
        save_json(run / "frozen_audit/config.json", {"config": {"training_seeds": seeds}})
        save_json(run / "frozen_audit/splits.json", {"plans": [{"name": name} for name in folds]})
        config = {"smoke": False, "config": {}, "folds": folds, "seeds": seeds,
            "source_sha256": {"fixture_code.py": sha(code)}, "model_candidates": {name: [{}] for name in models}}
        save_json(run / "config.json", config)
        completed = []
        for fold in folds:
            for seed in seeds:
                for model in models:
                    unit = run / fold / str(seed) / model
                    (unit / "test").mkdir(parents=True)
                    save_json(unit / "resources.json", {})
                    save_json(unit / "test/metrics.json", {})
                    # The gate checks finalization, not numerical contents. Full
                    # artifact/metric audit is a separate required workflow.
                    (unit / "test/predictions.npz").write_bytes(b"fixture")
                    completed.append({"fold": fold, "seed": seed, "model": model})
        save_json(run / "progress.json", {"complete": True, "completed": completed})
        return {"run": str(run), "config_sha256": sha(run / "config.json")}

    def test_selection_uses_cell_macro_source_errors_only(self):
        selected, scores = select_source_candidate(self.candidates(), {"a", "b"})
        self.assertEqual(selected["index"], 1)
        self.assertEqual(selected["source_validation_cell_macro_mape_percent"], 2)
        self.assertEqual(len(scores), 3)

    def test_target_id_cannot_enter_selection_even_at_secondary_origin(self):
        candidates = self.candidates()
        candidates[0]["by_cell"].append({"cell_id": "target", "origin": 20, "metrics": {"mape_percent": 0}})
        with self.assertRaisesRegex(ValueError, "non-source"):
            select_source_candidate(candidates, {"a", "b"})

    def test_missing_duplicate_or_nonfinite_source_errors_are_not_dropped(self):
        for mutation in ("missing", "duplicate", "nonfinite"):
            candidates = self.candidates()
            if mutation == "missing":
                candidates[0]["by_cell"].pop()
            elif mutation == "duplicate":
                candidates[0]["by_cell"].append(copy.deepcopy(candidates[0]["by_cell"][0]))
            else:
                candidates[0]["by_cell"][0]["metrics"]["mape_percent"] = np.inf
            with self.assertRaises(ValueError):
                select_source_candidate(candidates, {"a", "b"})

    def test_complete_finalization_gate_and_incomplete_progress(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            binding = self.phase_fixture(run)
            self.assertEqual(assert_main_complete([binding])[0]["finalized_units"], 400)
            progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
            progress["complete"] = False
            save_json(run / "progress.json", progress)
            with self.assertRaisesRegex(ValueError, "main experiment incomplete"):
                assert_main_complete([binding])

    def test_missing_outer_fold_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            binding = self.phase_fixture(run)
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            config["folds"].pop()
            save_json(run / "config.json", config)
            binding["config_sha256"] = sha(run / "config.json")
            with self.assertRaisesRegex(ValueError, "subset of frozen outer folds"):
                assert_main_complete([binding])

    def test_single_seed_or_missing_final_marker_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            binding = self.phase_fixture(run)
            marker = next(run.glob("MIT_fold1/*/*/resources.json"))
            marker.unlink()
            with self.assertRaisesRegex(ValueError, "unfinalized"):
                assert_main_complete([binding])
            save_json(marker, {})
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            config["seeds"] = config["seeds"][:1]
            save_json(run / "config.json", config)
            binding["config_sha256"] = sha(run / "config.json")
            with self.assertRaisesRegex(ValueError, "all frozen training seeds"):
                assert_main_complete([binding])

    def test_missing_required_models_cannot_be_complete(self):
        with self.assertRaisesRegex(ValueError, "required main models"):
            assert_main_complete([])


class NumericalAuditBindingTests(unittest.TestCase):
    """Toy fingerprints check the gate, not trained models or numeric accuracy."""

    def fixture(self, root):
        run = root / "main_run"
        run.mkdir()
        binding = SourceSelectionAndPhaseTests().phase_fixture(run)
        audit = root / "audits" / "snapshot"
        (audit / "source_snapshot").mkdir(parents=True)
        code = audit / "source_snapshot/fixture_auditor.py"
        code.write_text("# toy audit source, not a real numerical audit\n", encoding="utf-8")
        save_json(audit / "config.json", {"audit_contract_schema_version": 2,
            "source_sha256": {"fixture_auditor.py": sha(code)}})
        config = json.loads((run / "config.json").read_text())
        progress = json.loads((run / "progress.json").read_text())
        rows = []
        for item in progress["completed"]:
            unit = run / item["fold"] / str(item["seed"]) / item["model"]
            files = ["config.json", "selected.json", "validation_scores.json", "resources.json",
                "model.joblib", "test/metrics.json", "test/predictions.npz", "test/curves_origin100.png",
                "candidate_0/model.joblib", "candidate_0/validation/metrics.json", "candidate_0/validation/predictions.npz"]
            hashes = {}
            for name in files:
                path = unit / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(b"toy fingerprint fixture, not model weights")
                hashes[str(Path(name))] = sha(path)
            rows.append({"run": run.name, **item, "status": "passed static artifact/metric audit", "artifact_sha256": hashes})
        report = {"audit_contract_schema_version": 2, "failed_runs": 0, "failed_units": 0,
            "runs": [{"run": str(run), "expected_units": 400, "snapshot_completed_units": 400,
                "run_claimed_complete": True, "pending_units": [], "top_level_failure_present": False,
                "run_config_sha256": binding["config_sha256"], "progress_sha256_at_snapshot": sha(run / "progress.json"),
                "frozen_source_sha256": config["source_sha256"],
                "frozen_audit_sha256": {str(p.relative_to(run / "frozen_audit")): sha(p)
                    for p in (run / "frozen_audit").rglob("*") if p.is_file()}}], "units": rows}
        save_json(audit / "report.json", report)
        return run, audit, binding, report

    def check(self, root, binding):
        return assert_main_numerical_audits([binding], root / "audits")

    def test_full_current_audit_binding_passes_but_is_not_executable_replay(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, _, binding, _ = self.fixture(root)
            self.assertEqual(self.check(root, binding)[0]["audited_units"], 400)

    def test_full_finalization_without_numerical_report_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "main_run"
            run.mkdir()
            binding = SourceSelectionAndPhaseTests().phase_fixture(run)
            with self.assertRaisesRegex(ValueError, "current full numerical audit required"):
                self.check(root, binding)

    def test_changed_predictions_weights_or_frozen_audit_refused(self):
        for what in ("test/predictions.npz", "model.joblib", "frozen"):
            with self.subTest(what=what), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                run, _, binding, report = self.fixture(root)
                if what == "frozen":
                    path = run / "frozen_audit/config.json"
                    value = json.loads(path.read_text())
                    value["changed"] = True
                    save_json(path, value)
                else:
                    item = report["units"][0]
                    path = run / item["fold"] / str(item["seed"]) / item["model"] / what
                    path.write_bytes(b"changed fixture")
                with self.assertRaisesRegex(ValueError, "current full numerical audit required"):
                    self.check(root, binding)

    def test_missing_duplicate_or_failed_audit_unit_refused(self):
        for mutation in ("missing", "duplicate", "failed", "old"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                _, audit, binding, report = self.fixture(root)
                if mutation == "missing":
                    report["units"].pop()
                elif mutation == "duplicate":
                    report["units"].append(copy.deepcopy(report["units"][0]))
                elif mutation == "failed":
                    report["units"][0]["status"] = "failed artifact audit"
                else:
                    report.pop("audit_contract_schema_version")
                save_json(audit / "report.json", report)
                with self.assertRaisesRegex(ValueError, "current full numerical audit required"):
                    self.check(root, binding)

    def test_changed_auditor_or_escaping_artifact_path_refused(self):
        for mutation in ("source", "escape"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                _, audit, binding, report = self.fixture(root)
                if mutation == "source":
                    (audit / "source_snapshot/fixture_auditor.py").write_text("changed\n")
                else:
                    report["units"][0]["artifact_sha256"]["../../outside"] = "bad"
                    save_json(audit / "report.json", report)
                with self.assertRaisesRegex(ValueError, "current full numerical audit required"):
                    self.check(root, binding)


if __name__ == "__main__":
    unittest.main()
