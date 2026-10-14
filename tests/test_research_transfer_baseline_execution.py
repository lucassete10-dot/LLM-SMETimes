"""Real synthetic baseline fits/recovery; toy main bindings are NOT public evidence."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from battery_smetimes import research_transfer_baseline_execution as execution
from battery_smetimes.research_transfer_baselines import predict_transfer
from battery_smetimes.research_transfer_neural_recovery import exclusive_run_lock
from battery_smetimes.research_transfer_signals import RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.prepare_same_type_forecast import save_json
from scripts.train_soh_research_transfer_baselines_v5 import derive_baseline_source
from tests import test_research_transfer_execution as fixture


def scores(folder, report, arrays):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "metrics.json", report)
    np.savez_compressed(folder / "predictions.npz", **arrays)


class SharedBaselineExecutionTests(unittest.TestCase):
    def setUp(self):
        source = fixture.TransferFitTests()
        source.setUp()
        self.cells, self.plan = source.cells, source.plan
        self.plan["name"] = "MIT_to_NASA_synthetic"
        self.fold = {"name": "source_fold1", "source_train": ["s1", "s2"], "source_validation": ["s3"]}
        self.plan["source_selection_folds"] = [self.fold]
        self.parameters = {"alpha": 10.}
        self.config = {"smoke": True, "origins": [20, 50, 100],
            "baseline_source": {"binding": {"run": "toy"}, "tree_estimators": 48,
                "candidates": {"ridge": [self.parameters], "extra_trees": [{"min_samples_leaf": 2}]}}}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.unit = self.root / self.plan["name"] / "9" / "ridge"
        self.stage = self.unit / "candidate_0" / self.fold["name"]

    def train(self, destination=None, *, fold=True, model="ridge"):
        with patch("scripts.train_soh_research_v5.save_scores", side_effect=scores):
            return execution.train_baseline_stage(self.cells, self.plan, [20, 50, 100], model,
                self.parameters if model == "ridge" else {"min_samples_leaf": 2}, 9, True,
                self.stage if destination is None else destination, fold=self.fold if fold else None)

    def audit(self):
        return execution.audit_baseline_stage(self.stage, self.cells, self.plan, [20, 50, 100], "ridge", self.parameters, 9, True, fold=self.fold)

    def execute(self, *, resume, session):
        with exclusive_run_lock(self.root), patch("scripts.train_soh_research_v5.save_scores", side_effect=scores):
            return execution.execute_baseline_unit(self.root, self.config, self.plan, 9, "ridge", self.cells, resume=resume, session=session)

    @staticmethod
    def hashes(folder):
        return {str(p.relative_to(folder)): stream_sha(p) for p in folder.rglob("*") if p.is_file()}

    def test_source_fits_and_static_checks_do_not_access_target_signals_or_labels(self):
        before, _ = self.train()
        for key in self.plan["target_test"]:
            self.cells[key].update(soh=object(), label_cycle=object(), common=np.full((100, 11), np.inf))
        after, report = self.train(self.unit / "candidate_1" / self.fold["name"])
        np.testing.assert_array_equal(before["model"].coef_, after["model"].coef_)
        np.testing.assert_array_equal(before["scaler"].mean_, after["scaler"].mean_)
        np.testing.assert_array_equal(before["imputer"].statistics_, after["imputer"].statistics_)
        self.assertFalse(report["target_evaluated_during_training"])
        with patch("joblib.load", side_effect=AssertionError("static audit loaded")):
            self.audit()

    def test_completed_stage_reused_without_fitting_or_deserializing(self):
        self.train()
        original = self.hashes(self.stage)
        with patch.object(execution, "fit_transfer", side_effect=AssertionError("refit")), \
             patch("joblib.load", side_effect=AssertionError("candidate load")):
            bundle, report = execution.recover_baseline_stage(self.cells, self.plan, [20, 50, 100], "ridge", self.parameters,
                9, True, self.stage, run=self.root, session="resume", resume=True, fold=self.fold)
        self.assertIsNone(bundle)
        self.assertTrue(report["complete"])
        self.assertEqual(self.hashes(self.stage), original)

    def test_interrupted_export_preserved_and_same_seed_model_restarted(self):
        original_dump = execution.joblib.dump
        def fail_dump(bundle, destination):
            original_dump(bundle, destination)
            raise RuntimeError("injected export interruption")
        with patch.object(execution.joblib, "dump", side_effect=fail_dump):
            with self.assertRaisesRegex(RuntimeError, "export interruption"):
                self.train()
        partial = self.hashes(self.stage)
        with patch("scripts.train_soh_research_v5.save_scores", side_effect=scores):
            actual, _ = execution.recover_baseline_stage(self.cells, self.plan, [20, 50, 100], "ridge", self.parameters,
                9, True, self.stage, run=self.root, session="resume", resume=True, fold=self.fold, need_bundle=True)
        archived = self.root / "interrupted_attempts/resume" / self.stage.relative_to(self.root)
        self.assertEqual(self.hashes(archived), partial)
        expected, _ = self.train(self.root / "independent_reference")
        np.testing.assert_array_equal(actual["model"].coef_, expected["model"].coef_)
        self.assertEqual(actual["seed"], 9)

    def test_bad_complete_weights_or_config_do_not_trigger_retraining(self):
        self.train()
        for name in ("model.joblib", "model.json", "resources.json", "config.json"):
            path = self.stage / name
            original = path.read_bytes()
            path.write_bytes(b"changed")
            try:
                with patch.object(execution, "fit_transfer", side_effect=AssertionError("refit")), \
                     patch("joblib.load", side_effect=AssertionError("changed bytes loaded")):
                    with self.assertRaises((ValueError, json.JSONDecodeError)):
                        execution.recover_baseline_stage(self.cells, self.plan, [20, 50, 100], "ridge", self.parameters,
                            9, True, self.stage, run=self.root, session="resume", resume=True, fold=self.fold)
            finally:
                path.write_bytes(original)
        self.assertFalse((self.root / "interrupted_attempts").exists())

    def test_complete_unit_and_high_error_target_retained_and_reused(self):
        self.cells["t2"]["soh"][:] = .01
        actual = self.execute(resume=False, session="first")
        self.assertGreater(actual["target_summary"]["100"]["cell_macro_mape_percent"], 6)
        self.assertEqual(actual["target_summary"]["100"]["cells"], 2)
        original = self.hashes(self.unit)
        with patch.object(execution, "fit_transfer", side_effect=AssertionError("refit")), \
             patch.object(execution, "evaluate_transfer", side_effect=AssertionError("re-evaluation")), \
             patch("joblib.load", side_effect=AssertionError("unneeded load")):
            recovered = self.execute(resume=True, session="second")
        self.assertEqual(recovered, actual)
        self.assertEqual(self.hashes(self.unit), original)

    def test_all_source_refit_reloaded_after_target_failure_without_refitting(self):
        original = execution.evaluate_transfer
        def fail_target(bundle, cells, ids):
            if ids == self.plan["target_test"]:
                raise RuntimeError("target failure")
            return original(bundle, cells, ids)
        with patch.object(execution, "evaluate_transfer", side_effect=fail_target):
            with self.assertRaisesRegex(RuntimeError, "target failure"):
                self.execute(resume=False, session="first")
        refit = self.hashes(self.unit / "source_refit")
        with patch.object(execution, "fit_transfer", side_effect=AssertionError("refit")):
            self.execute(resume=True, session="second")
        self.assertEqual(self.hashes(self.unit / "source_refit"), refit)

    def test_partial_target_save_archived_without_changing_source_selection(self):
        def interrupted(folder, report, arrays):
            scores(folder, report, arrays)
            if folder.name == "target_test":
                raise RuntimeError("target save failure")
        with patch("scripts.train_soh_research_v5.save_scores", side_effect=interrupted), exclusive_run_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "save failure"):
                execution.execute_baseline_unit(self.root, self.config, self.plan, 9, "ridge", self.cells, resume=False, session="first")
        partial = self.hashes(self.unit / "target_test")
        selected = stream_sha(self.unit / "selected.json")
        with patch.object(execution, "fit_transfer", side_effect=AssertionError("refit")):
            self.execute(resume=True, session="second")
        archive = self.root / "interrupted_attempts/second" / (self.unit / "target_test").relative_to(self.root)
        self.assertEqual(self.hashes(archive), partial)
        self.assertEqual(stream_sha(self.unit / "selected.json"), selected)

    def test_missing_primary_source_validation_cell_cannot_be_hidden(self):
        self.cells["s3"]["reference_established_source_cycle"] = 120
        with self.assertRaisesRegex(ValueError, "every primary"):
            self.train()
        self.assertTrue((self.stage / "failure.json").is_file())
        self.assertFalse((self.stage / "report.json").is_file())

    def test_actual_forest_saved_estimator_and_ridge_load_without_fit(self):
        for model in ("ridge", "extra_trees"):
            folder = self.root / model
            before, _ = self.train(folder, fold=False, model=model)
            with patch.object(execution, "fit_transfer", side_effect=AssertionError("refit")):
                restored = execution.load_baseline_checkpoint(folder)
            np.testing.assert_allclose(predict_transfer(before, self.cells["t1"], 100, [110, 120], mode=RESEARCH_MODE),
                predict_transfer(restored, self.cells["t1"], 100, [110, 120], mode=RESEARCH_MODE), atol=1e-10, rtol=1e-10)
            if model == "extra_trees":
                self.assertEqual(len(restored["model"].estimators_), 48)
                self.assertIsNone(restored["scaler"])


class SharedBaselineBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root / "source_snapshot/source.py"
        path.parent.mkdir(parents=True)
        path.write_text("# toy native source only\n")
        self.inputs = {"training_seeds": [1, 2, 3, 4, 5], "origins_source_cycles": [20, 50, 100]}
        self.config = {"source_sha256": {"source.py": stream_sha(path)}, "smoke": False, "tree_estimators": 160,
            "seeds": self.inputs["training_seeds"], "config": {"origins_source_cycles": self.inputs["origins_source_cycles"],
                "initial_candidates": {"ridge": [{"alpha": a} for a in (.1, 10., 100.)],
                    "extra_trees": [{"min_samples_leaf": a} for a in (2, 4, 16)]}}}
        save_json(self.root / "config.json", self.config)
        self.binding = {"run": str(self.root), "config_sha256": stream_sha(self.root / "config.json")}

    def test_three_original_candidates_full_one_explicit_smoke_and_original_tree_count(self):
        full = derive_baseline_source([self.binding], self.inputs, smoke=False)
        smoke = derive_baseline_source([self.binding], self.inputs, smoke=True)
        self.assertEqual(full["tree_estimators"], 160)
        self.assertEqual(smoke["tree_estimators"], 48)
        for name in ("ridge", "extra_trees"):
            self.assertEqual(len(full["candidates"][name]), 3)
            self.assertEqual(len(smoke["candidates"][name]), 1)
            self.assertEqual(smoke["original_candidates"][name], full["candidates"][name])

    def test_missing_ambiguous_changed_budget_seed_or_source_rejected(self):
        for bindings in ([], [self.binding, self.binding]):
            with self.assertRaises(ValueError):
                derive_baseline_source(bindings, self.inputs, smoke=False)
        for kind in ("smoke", "seed", "origin", "trees", "budget", "binding"):
            config, binding = copy.deepcopy(self.config), copy.deepcopy(self.binding)
            if kind == "smoke":
                config["smoke"] = True
            elif kind == "seed":
                config["seeds"] = [1]
            elif kind == "origin":
                config["config"]["origins_source_cycles"] = [100]
            elif kind == "trees":
                config["tree_estimators"] = 48
            else:
                config["config"]["initial_candidates"]["ridge"] = [{"alpha": 1}]
            save_json(self.root / "config.json", config)
            if kind != "binding":
                binding["config_sha256"] = stream_sha(self.root / "config.json")
            with self.assertRaises(ValueError):
                derive_baseline_source([binding], self.inputs, smoke=False)

    def test_real_saved_direct_worker_bootstrap_refuses_prepare_only_without_fitting(self):
        from scripts.train_soh_research_transfer_baselines_v5 import SOURCES
        workspace = Path(__file__).resolve().parents[1]
        parent = workspace / "experiments/soh_research_v5/transfer_baseline_runs"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="bootstrap_test_", dir=parent) as name:
            run = Path(name)
            hashes = {}
            for source in SOURCES:
                path = run / "source_snapshot" / source
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(workspace / source, path)
                hashes[source] = stream_sha(path)
            save_json(run / "config.json", {"source_sha256": hashes, "prepare_only": True})
            save_json(run / "progress.json", {"completed": [], "model_training_started": False})
            process = subprocess.run([sys.executable, str(run / "source_snapshot/scripts/train_soh_research_transfer_baselines_v5.py"),
                "--run", str(run)], cwd=workspace, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("prepared-only contract cannot be repurposed", process.stderr)
            self.assertFalse(json.loads((run / "failure.json").read_text())["state"]["model_training_started"])
            self.assertFalse(list(run.rglob("model.joblib")))


if __name__ == "__main__":
    unittest.main()
