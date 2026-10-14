"""Executable synthetic baseline replay, not public-data accuracy evidence."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from battery_smetimes import research_transfer_baseline_replay as replay
from battery_smetimes.research_transfer_baseline_execution import load_baseline_checkpoint
from battery_smetimes.research_transfer_baselines import evaluate_transfer
from battery_smetimes.research_transfer_replay_checks import causal_predictions
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.prepare_same_type_forecast import save_json
from tests import test_research_transfer_baseline_execution as fixture_module
from tests import test_research_transfer_neural_replay as parent_fixture_module


class BaselineReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.SharedBaselineExecutionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.cells, self.plan = self.fixture.root, self.fixture.cells, self.fixture.plan
        self.cells["t2"]["soh"][:] = .01  # Valid hard target must remain.
        self.folder = self.root / "source_refit"
        self.bundle, _ = self.fixture.train(self.folder, fold=False)
        self.report, self.saved = evaluate_transfer(self.bundle, self.cells, self.plan["target_test"])

    def test_actual_ridge_reloads_and_replays_all_targets_without_fitting(self):
        with patch("sklearn.linear_model.Ridge.fit", side_effect=AssertionError("replay fitted")):
            restored = load_baseline_checkpoint(self.folder)
            rows, arrays = replay.replay_unit(restored, self.cells, self.plan, self.saved)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["passed"] and r["atol"] == 1e-10 for r in rows))
        self.assertTrue(all(r["causal_max_absolute_difference"] == 0 for r in rows))
        self.assertGreater(self.report["summary"]["100"]["cell_macro_mape_percent"], 6)
        self.assertEqual({r["cell_id"] for r in rows}, {"t1", "t2"})
        for key, value in arrays.items():
            if key.endswith("__intact"):
                prefix = key[:-len("intact")]
                np.testing.assert_array_equal(value, arrays[prefix+"prefix_only"])
                np.testing.assert_array_equal(value, arrays[prefix+"poisoned_future"])

    def test_actual_forest_original_runtime_match_and_serial_causal_control(self):
        bundle, _ = self.fixture.train(self.root / "forest_refit", fold=False, model="extra_trees")
        _, saved = evaluate_transfer(bundle, self.cells, self.plan["target_test"])
        restored = load_baseline_checkpoint(self.root / "forest_refit")
        with patch("sklearn.ensemble.ExtraTreesRegressor.fit", side_effect=AssertionError("replay fitted")):
            rows, _ = replay.replay_unit(restored, self.cells, self.plan, saved)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["passed"] and r["causal_max_absolute_difference"] == 0 for r in rows))
        self.assertTrue(all(r["causal_control"]["original_n_jobs"] == 4 for r in rows))
        self.assertTrue(all(r["causal_control"]["serial_tree_accumulation_control"] for r in rows))
        self.assertEqual(restored["model"].n_jobs, 4)

    def test_causal_control_restores_forest_runtime_even_on_failure(self):
        class Forest:
            estimators_ = [object()]
            n_jobs = 4
        model = Forest()
        def failing(*args, **kwargs):
            self.assertEqual(model.n_jobs, 1)
            raise RuntimeError("injected prediction failure")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            causal_predictions(failing, {"model": model}, {}, {}, {}, 20, np.array([21, 22]), np.array([.8, .8]))
        self.assertEqual(model.n_jobs, 4)

    def test_changed_saved_prediction_rejected_with_failed_evidence(self):
        saved = {key: value.copy() for key, value in self.saved.items()}
        saved["t1__20__prediction"] += .01
        recorded = []
        with self.assertRaisesRegex(ValueError, "failed saved-prediction"):
            replay.replay_unit(self.bundle, self.cells, self.plan, saved,
                progress_callback=lambda rows, arrays: recorded.extend(copy.deepcopy(rows[-1:])))
        self.assertFalse(recorded[-1]["passed"])
        self.assertTrue(recorded[-1]["same_device_no_future_invariance"])
        self.assertEqual(recorded[-1]["atol"], 1e-10)

    def test_label_future_and_length_leaks_rejected_at_n100(self):
        predictor = replay.predict_transfer
        for kind in ("label", "length", "future"):
            def leaking(bundle, cell, origin, future, **kwargs):
                values = predictor(bundle, cell, origin, future, **kwargs)
                delta = (.01 if "soh" in cell else 0) if kind == "label" else (
                    .001*len(cell["cycle"]) if kind == "length" else (.01 if np.isinf(cell["common"]).any() else 0))
                return values+delta
            with self.subTest(kind=kind), patch.object(replay, "predict_transfer", side_effect=leaking):
                with self.assertRaisesRegex(ValueError, "causal replay"):
                    replay.replay_unit(self.bundle, self.cells, self.plan, self.saved, origins=[100])

    def test_coverage_gaps_preserved_and_empty_checks_not_success(self):
        self.cells["t2"]["reference_established_source_cycle"] = 65
        _, saved = evaluate_transfer(self.bundle, self.cells, self.plan["target_test"])
        rows, _ = replay.replay_unit(self.bundle, self.cells, self.plan, saved)
        self.assertEqual(sum(r.get("passed", False) for r in rows), 4)
        self.assertEqual(sum("status" in r for r in rows), 2)
        with self.assertRaisesRegex(ValueError, "no originally evaluated"):
            replay.replay_unit(self.bundle, self.cells, {**self.plan, "target_test": ["t2"]}, saved, origins=[20, 50])

    def test_parent_and_weights_changes_rejected_before_deserialization(self):
        self.fixture.execute(resume=False, session="synthetic")
        config = {**self.fixture.config, "source_sha256": {}, "seeds": [9], "models": ["ridge"], "device": "cpu"}
        for name, value in (("config.json", config), ("progress.json", {"toy": True}),
                            ("splits.json", {"directions": [self.plan]}), ("static_run_audit.json", {"toy": True})):
            save_json(self.root / name, value)
        evidence = {key: stream_sha(self.root / name) for name, key in (("config.json", "run_config_sha256"),
            ("progress.json", "progress_sha256"), ("splits.json", "splits_sha256"), ("static_run_audit.json", "static_run_audit_sha256"))}
        verified = {"run": self.root, "config": config, "plans": [self.plan], "cells": self.cells, "evidence": evidence}
        # Actual unit checks; synthetic parent is not a verified public run.
        with patch.object(replay, "verify_model_resolution", return_value={"toy": True}), \
             patch("sklearn.linear_model.Ridge.fit", side_effect=AssertionError("replay fitted")):
            _, loaded = replay.load_verified_unit(verified, self.plan, 9, "ridge")
        self.assertTrue(loaded["checkpoint"]["no_fit"])
        for name in ("config.json", "progress.json", "splits.json", "static_run_audit.json"):
            path = self.root / name
            before = path.read_bytes()
            save_json(path, {"changed": True})
            try:
                with patch("joblib.load", side_effect=AssertionError("bad parent deserialized")):
                    with self.assertRaisesRegex(ValueError, "parent evidence changed"):
                        replay.load_verified_unit(verified, self.plan, 9, "ridge")
            finally:
                path.write_bytes(before)  # Exact owned temporary fixture only.
        with patch("joblib.load", side_effect=AssertionError("CUDA baseline deserialized")):
            with self.assertRaisesRegex(ValueError, "CPU backend"):
                replay.load_verified_unit(verified, self.plan, 9, "ridge", device="cuda")
        save_json(self.fixture.unit / "source_refit/model.json", {"changed": True})
        with patch.object(replay, "verify_model_resolution", return_value={"toy": True}), \
             patch("joblib.load", side_effect=AssertionError("bad weights deserialized")):
            with self.assertRaises(ValueError):
                replay.load_verified_unit(verified, self.plan, 9, "ridge")

    def test_fresh_interpreter_uses_actual_frozen_sources_without_torch_or_fit(self):
        from scripts.train_soh_research_transfer_baselines_v5 import SOURCES
        workspace = Path(__file__).resolve().parents[1]
        hashes = {}
        for name in SOURCES:
            path = self.root / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workspace / name, path)
            hashes[name] = stream_sha(path)
        save_json(self.root / "config.json", {"source_sha256": hashes, "plan": self.plan})
        descriptors, arrays = {}, {}
        for key, cell in self.cells.items():
            descriptors[key] = {name: value for name, value in cell.items() if not isinstance(value, np.ndarray)}
            arrays.update({key+"__"+name: value for name, value in cell.items() if isinstance(value, np.ndarray)})
        save_json(self.root / "synthetic_cells.json", descriptors)
        np.savez_compressed(self.root / "synthetic_cells.npz", **arrays)
        np.savez_compressed(self.root / "saved_predictions.npz", **self.saved)
        code = """
import json,sys
from pathlib import Path
run=Path(sys.argv[1]); sys.path.insert(0,str(run/'source_snapshot'))
import battery_smetimes,scripts
battery_smetimes.__path__=[str(run/'source_snapshot/battery_smetimes'),*battery_smetimes.__path__]
scripts.__path__=[str(run/'source_snapshot/scripts'),*scripts.__path__]
from battery_smetimes.research_transfer_baseline_replay import verify_model_resolution,replay_unit
from battery_smetimes.research_transfer_baseline_execution import load_baseline_checkpoint
from battery_smetimes.research_transfer_neural_recovery import verify_root_utility_resolution
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
def no_fit(*args,**kwargs): raise AssertionError('replay fitted')
Ridge.fit=no_fit; ExtraTreesRegressor.fit=no_fit
config=json.loads((run/'config.json').read_text())
paths=verify_model_resolution(run,config); root_paths=verify_root_utility_resolution(run,config)
cells=json.loads((run/'synthetic_cells.json').read_text())
with np.load(run/'synthetic_cells.npz',allow_pickle=False) as arrays:
 for name in arrays.files:
  key,field=name.split('__'); cells[key][field]=arrays[name]
bundle=load_baseline_checkpoint(run/'source_refit')
with np.load(run/'saved_predictions.npz',allow_pickle=False) as saved:
 rows,_=replay_unit(bundle,cells,config['plan'],saved)
print(json.dumps({'checks':len(rows),'passed':all(r['passed'] for r in rows),'source_paths':paths,'root_paths':root_paths,'torch_imported':'torch' in sys.modules}))
"""
        result = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-W", "error::UserWarning", "-c", code, str(self.root)],
            cwd=workspace, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout)
        self.assertEqual(evidence["checks"], 6)
        self.assertTrue(evidence["passed"])
        self.assertFalse(evidence["torch_imported"])
        self.assertEqual(len(evidence["root_paths"]), 5)
        self.assertTrue(all(Path(path).is_relative_to(self.root / "source_snapshot") for path in evidence["source_paths"].values()))


class BaselineReplayParentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = parent_fixture_module.SharedReplayParentTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run, self.config = self.fixture.run, self.fixture.config

    def test_partial_prepared_failed_or_changed_source_cannot_reach_contract_or_weights(self):
        for kind in ("prepare", "partial", "not_started", "failure", "source"):
            config, state = copy.deepcopy(self.config), copy.deepcopy(self.fixture.state)
            if kind == "prepare":
                config["prepare_only"] = True
            elif kind == "partial":
                state["complete"] = False
            elif kind == "not_started":
                state["model_training_started"] = False
            elif kind == "failure":
                save_json(self.run / "failure.json", {"injected": True})
            else:
                config["source_sha256"]["toy_source.py"] = "wrong"
            save_json(self.run / "config.json", config)
            save_json(self.run / "progress.json", state)
            with self.subTest(kind=kind), \
                 patch.object(replay, "verify_baseline_contract", side_effect=AssertionError("contract reached too early")), \
                 patch("joblib.load", side_effect=AssertionError("weights loaded too early")):
                with self.assertRaises(ValueError):
                    replay.verify_replay_run(self.run)
            if (self.run / "failure.json").exists():
                (self.run / "failure.json").unlink()  # Exact owned injected fixture.

    def test_parent_uses_original_main_numerical_gate_and_full_static_evidence_without_loading(self):
        config = {**self.config, "original_model_sha256": {"toy": "toy"}}
        with patch.object(replay, "verify_baseline_contract", return_value=(config, [])), \
             patch.object(replay, "assert_main_complete", return_value={"toy_main": True}), \
             patch.object(replay, "assert_main_numerical_audits", return_value={"toy_numeric": True}), \
             patch.object(replay, "verify_root_utility_resolution", return_value={"toy_roots": True}), \
             patch.object(replay, "load_shared_cells", return_value={}), \
             patch.object(replay, "audit_baseline_run", return_value={"finalized_units": 40, "toy": True}), \
             patch("joblib.load", side_effect=AssertionError("verification loaded weights")):
            verified = replay.verify_replay_run(self.run)
        self.assertFalse(verified["evidence"]["weights_loaded"])
        self.assertEqual(verified["evidence"]["main_numerical"], {"toy_numeric": True})
        with self.assertRaisesRegex(ValueError, "outside frozen source"):
            replay.verify_model_resolution(self.run, config)


if __name__ == "__main__":
    unittest.main()
