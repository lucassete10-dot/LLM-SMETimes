"""Synthetic fitted-unit audit and toy smoke-prerequisite logic, not public accuracy."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from battery_smetimes.research_transfer_neural_audit import audit_shared_neural_unit, assert_shared_neural_smoke, audit_shared_neural_run
from battery_smetimes.research_transfer_neural_training import train_shared_source_unit
from battery_smetimes.research_transfer_baselines import evaluate_transfer
from battery_smetimes.research_transfer_sequences import predict_transfer_sequence
from battery_smetimes.research_phase_gate import select_source_candidate
from scripts.prepare_same_type_forecast import save_json, sha
from tests import test_research_transfer_neural_training as training_fixture


def json_at(path):
    return json.loads(path.read_text())


class SharedUnitAuditTests(unittest.TestCase):
    def setUp(self):
        fixture = training_fixture.SharedNeuralTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.root, self.cells, self.plan, self.fold, self.parameters = fixture.root, fixture.cells, fixture.plan, fixture.fold, fixture.params
        self.settings = {**fixture.config, "candidates": [self.parameters], "smoke_epochs": 2}
        self.source = {"configuration": self.settings, "checkpoint_files_sha256": {}, "binding": {"run": "toy_source"}}
        self.run_config = {"smoke": True, "origins": [20, 50, 100], "model_sources": {"gru": self.source}}
        self.unit = self.root / "unit"
        self.unit.mkdir()
        save_json(self.unit / "config.json", {"plan": self.plan, "seed": 9, "model": "gru", "source": self.source, "folds": [self.fold]})
        _, fold_report = train_shared_source_unit(self.cells, self.plan, [20, 50, 100], "gru", self.parameters, 9, "cpu",
            self.settings, self.unit / "candidate_0" / self.fold["name"], source_fold=self.fold)
        candidates = [{"index": 0, "parameters": self.parameters, "by_cell": fold_report["source_validation_by_cell"]}]
        selected, scores = select_source_candidate(candidates, self.fold["source_validation"])
        selected = {**selected, "fixed_refit_epochs": fold_report["selected_epochs"], "source_fold_selected_epochs": [fold_report["selected_epochs"]]}
        save_json(self.unit / "source_candidates.json", candidates)
        save_json(self.unit / "selected.json", selected)
        save_json(self.unit / "source_validation_scores.json", scores)
        self.bundle, _ = train_shared_source_unit(self.cells, self.plan, [20, 50, 100], "gru", self.parameters, 9, "cpu", self.settings,
            self.unit / "source_refit", fixed_refit_epochs=selected["fixed_refit_epochs"])
        # A high-error target stays in the evaluated population.
        self.cells["t2"]["soh"][:] = .01
        self.target, self.arrays = evaluate_transfer(self.bundle, self.cells, self.plan["target_test"], predictor=predict_transfer_sequence)
        (self.unit / "target_test").mkdir()
        save_json(self.unit / "target_test/metrics.json", self.target)
        np.savez_compressed(self.unit / "target_test/predictions.npz", **self.arrays)
        save_json(self.unit / "target_evaluation_contract.json", {"target_calibration": [], "target_labels_used_for_selection": False,
            "physical_families_preserved": True, "not_same_type_deployment": True})

    def audit(self):
        return audit_shared_neural_unit(self.unit, self.run_config, self.plan, 9, "gru", self.cells)

    def test_recompute_all_source_target_metrics_and_do_not_gate_on_target_error(self):
        with patch("joblib.load", side_effect=AssertionError("static audit must not deserialize")):
            report = self.audit()
        self.assertEqual(report["target_cells"], 2)
        self.assertGreater(report["target_summary"]["100"]["cell_macro_mape_percent"], 6)
        self.assertEqual(report["candidate_count"], 1)
        self.assertIn(str(Path("source_refit/model.pt")), report["artifact_sha256"])
        save_json(self.unit / "static_unit_audit.json", report)
        self.assertEqual(self.audit(), report)  # audit excludes its own recursive digest

    def test_metric_grid_weights_source_candidate_epoch_and_identity_tampering_rejected(self):
        for kind in ("metric", "grid", "weight", "epoch", "family", "budget", "candidate"):
            with self.subTest(kind=kind):
                if kind == "grid":
                    path = self.unit / "target_test/predictions.npz"
                    original = path.read_bytes()
                    arrays = {key: value.copy() for key, value in self.arrays.items()}
                    arrays["t1__100__cycle"][0] += 1
                    np.savez_compressed(path, **arrays)
                elif kind == "weight":
                    path = self.unit / "source_refit/model.pt"
                    original = path.read_bytes()
                    path.write_bytes(b"changed")
                else:
                    path = self.unit / {"metric": "target_test/metrics.json", "epoch": "selected.json",
                        "family": "source_refit/model.json", "budget": "config.json", "candidate": "source_candidates.json"}[kind]
                    original = path.read_bytes()
                    value = json_at(path)
                    if kind == "metric":
                        value["by_cell"][0]["metrics"]["mape_percent"] += 1
                    elif kind == "epoch":
                        value["fixed_refit_epochs"] += 1
                    elif kind == "family":
                        value["allowed_cells"]["t1"]["physical_family"] = "source-family"
                    elif kind == "budget":
                        value["source"]["configuration"]["candidates"] = []
                    else:
                        value[0]["by_cell"][0]["cell_id"] = "t1"
                    save_json(path, value)
                try:
                    with self.assertRaises(ValueError):
                        self.audit()
                finally:
                    path.write_bytes(original)

    def test_valid_hard_cell_cannot_be_turned_into_self_consistent_skip(self):
        # Remove its arrays and recompute the apparently clean summaries. Only
        # independent eligibility replay reveals this score-based deletion.
        report = copy.deepcopy(self.target)
        removed = [r for r in report["by_cell"] if r["cell_id"] == "t2"]
        report["by_cell"] = [r for r in report["by_cell"] if r["cell_id"] != "t2"]
        report["skipped"].extend({"cell_id": "t2", "origin": r["origin"], "reason": "insufficient shared visible history"} for r in removed)
        for origin in (20, 50, 100):
            metric = next(r["metrics"]["mape_percent"] for r in report["by_cell"] if r["origin"] == origin)
            report["summary"][str(origin)] = {"cells": 1, "cell_macro_mape_percent": metric, "fraction_cells_mape_le_6": float(metric <= 6)}
        save_json(self.unit / "target_test/metrics.json", report)
        np.savez_compressed(self.unit / "target_test/predictions.npz", **{k: v for k, v in self.arrays.items() if k.startswith("t1__")})
        with self.assertRaisesRegex(ValueError, "eligibility"):
            self.audit()


class SharedSmokeEvidenceTests(unittest.TestCase):
    """Toy eight-unit ledgers test the consumer; NOT trained smoke evidence."""
    def fixture(self, root):
        smoke = root / "smoke"
        (smoke / "source_snapshot").mkdir(parents=True)
        code = smoke / "source_snapshot/source.py"
        code.write_text("# toy snapshot only\n")
        models = ["gru", "statistics_mlp", "llm_e1", "llm_e2"]
        source = {"configuration": {"candidates": [{"p": n} for n in (1, 2, 3)], "max_epochs": 60, "smoke_epochs": 2}}
        full = {"prepared": "toy_inputs", "prepared_config_sha256": "config", "prepared_manifest_sha256": "manifest",
            "prepared_splits_sha256": "splits", "main_run_bindings": [], "source_sha256": {"source.py": sha(code)},
            "origins": [20, 50, 100], "device": "cpu", "models": models, "input_schema": "toy", "feature_names": [],
            "packages": {}, "seeds": [1, 2, 3, 4, 5], "model_sources": {m: copy.deepcopy(source) for m in models},
            "target_calibration": [], "not_same_type_deployment": True, "global_summary_writer": False}
        config = {**copy.deepcopy(full), "smoke": True, "prepare_only": False, "seeds": [1]}
        for settings in config["model_sources"].values():
            settings["configuration"]["candidates"] = settings["configuration"]["candidates"][:1]
            settings["configuration"]["max_epochs"] = 2
        plans = [{"name": direction} for direction in ("MIT_to_XJTU", "XJTU_to_MIT")]
        completed = []
        for plan in plans:
            for model in models:
                unit = smoke / plan["name"] / "1" / model
                unit.mkdir(parents=True)
                save_json(unit / "static_unit_audit.json", {"toy_audit": True})
                completed.append({"direction": plan["name"], "seed": 1, "model": model})
        save_json(smoke / "config.json", config)
        save_json(smoke / "splits.json", {"directions": plans})
        save_json(smoke / "static_run_audit.json", {"toy_run_audit": True})
        save_json(smoke / "progress.json", {"complete": True, "model_training_started": True, "completed": completed, "expected_final_model_units": 8})
        return full, plans, smoke

    def test_complete_same_source_evidence_accepted_no_accuracy_threshold(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            full, plans, smoke = self.fixture(root)
            with patch("battery_smetimes.research_transfer_neural_audit.audit_shared_neural_unit", return_value={"toy_audit": True}), \
                 patch("battery_smetimes.research_transfer_neural_audit.audit_shared_neural_run", return_value={"toy_run_audit": True}):
                report = assert_shared_neural_smoke(root, full, plans, {})
            self.assertEqual(report["run"], str(smoke.resolve()))
            self.assertEqual(len(report["units"]), 8)
            self.assertIn("no target-score threshold", report["scope"])

    def test_prepare_partial_failed_stale_or_wrong_source_evidence_rejected(self):
        for kind in ("prepare", "partial", "failure", "stale", "source", "input", "budget", "run_audit", "transient"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                full, plans, smoke = self.fixture(root)
                config, state = json_at(smoke / "config.json"), json_at(smoke / "progress.json")
                if kind == "prepare":
                    config["prepare_only"] = True
                elif kind == "partial":
                    state["completed"].pop()
                elif kind == "failure":
                    save_json(smoke / "failure.json", {})
                elif kind == "source":
                    (smoke / "source_snapshot/source.py").write_text("changed")
                elif kind == "input":
                    config["input_schema"] = "other"
                elif kind == "budget":
                    config["model_sources"]["gru"]["configuration"]["max_epochs"] = 99
                elif kind == "run_audit":
                    save_json(smoke / "static_run_audit.json", {"toy_run_audit": False})
                save_json(smoke / "config.json", config)
                save_json(smoke / "progress.json", state)
                if kind == "transient":
                    (smoke / "progress.json").write_text("{partial")
                current = {"toy_audit": kind != "stale"}
                with patch("battery_smetimes.research_transfer_neural_audit.audit_shared_neural_unit", return_value=current), \
                     patch("battery_smetimes.research_transfer_neural_audit.audit_shared_neural_run", return_value={"toy_run_audit": True}):
                    with self.assertRaisesRegex(ValueError, "completed audited"):
                        assert_shared_neural_smoke(root, full, plans, {})


class SharedRunGridTests(unittest.TestCase):
    """Only whole-grid/binding logic; unit numerics are mocked explicitly."""
    def fixture(self, root, smoke=False):
        full, plans, run = SharedSmokeEvidenceTests().fixture(root)
        config = json_at(run / "config.json") if smoke else {**full, "smoke": False, "prepare_only": False}
        prepared = root / "prepared"
        (prepared / "source_snapshot").mkdir(parents=True)
        code = prepared / "source_snapshot/input.py"
        code.write_text("# prepared-binding fixture, no real battery inputs\n")
        inputs = {"training_seeds": full["seeds"], "origins_source_cycles": full["origins"], "input_schema": full["input_schema"],
            "feature_names": full["feature_names"], "source_sha256": {"input.py": sha(code)}}
        save_json(prepared / "config.json", inputs)
        save_json(prepared / "manifest.json", {"toy_only": True})
        save_json(prepared / "splits.json", {"directions": plans})
        config.update(prepared=str(prepared), prepared_config_sha256=sha(prepared / "config.json"),
            prepared_manifest_sha256=sha(prepared / "manifest.json"), prepared_splits_sha256=sha(prepared / "splits.json"))
        save_json(run / "config.json", config)
        completed = []
        for plan in plans:
            for seed in config["seeds"]:
                for model in config["models"]:
                    unit = run / plan["name"] / str(seed) / model
                    unit.mkdir(parents=True, exist_ok=True)
                    save_json(unit / "static_unit_audit.json", {"toy_audit": True, "target_summary": {}})
                    completed.append({"direction": plan["name"], "seed": seed, "model": model})
        save_json(run / "progress.json", {"complete": True, "model_training_started": True, "completed": completed,
            "expected_final_model_units": len(completed)})
        return run, config

    def check(self, run, config, **kwargs):
        with patch("scripts.train_soh_research_transfer_neural_v5.derive_model_sources", return_value=config["model_sources"]), \
             patch("battery_smetimes.research_transfer_neural_audit.audit_shared_neural_unit", return_value={"toy_audit": True, "target_summary": {}}):
            return audit_shared_neural_run(run, {}, **kwargs)

    def test_all_eight_smoke_or_forty_full_members_required(self):
        for smoke, count in ((True, 8), (False, 40)):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory() as name:
                run, config = self.fixture(Path(name), smoke)
                report = self.check(run, config)
                self.assertEqual(report["finalized_units"], count)
                self.assertEqual(len(report["units"]), count)
                self.assertEqual(len(report["completed_grid"]), count)
                state = json_at(run / "progress.json")
                state["completed"].pop()
                save_json(run / "progress.json", state)
                with self.assertRaisesRegex(ValueError, "complete model"):
                    self.check(run, config)

    def test_before_flag_finalization_is_explicit_and_stale_unit_cannot_pass(self):
        with tempfile.TemporaryDirectory() as name:
            run, config = self.fixture(Path(name), True)
            state = json_at(run / "progress.json")
            state["complete"] = False
            save_json(run / "progress.json", state)
            with self.assertRaisesRegex(ValueError, "complete fitted"):
                self.check(run, config)
            before = self.check(run, config, require_complete=False)
            state["complete"] = True
            save_json(run / "progress.json", state)
            self.assertEqual(self.check(run, config), before)
            next(run.glob("*/*/*/static_unit_audit.json")).write_text('{}')
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                self.check(run, config)


if __name__ == "__main__":
    unittest.main()
