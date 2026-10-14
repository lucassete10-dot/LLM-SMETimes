"""Actual synthetic CPU replay; toy parent contracts are NOT public run evidence."""

import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from battery_smetimes import research_transfer_neural_replay as replay
from battery_smetimes.research_transfer_neural_training import train_shared_source_unit, load_shared_neural_checkpoint
from battery_smetimes.research_transfer_baselines import evaluate_transfer
from battery_smetimes.research_transfer_sequences import predict_transfer_sequence
from scripts.prepare_same_type_forecast import save_json
from battery_smetimes.research_transfer_vi120 import stream_sha
from tests import test_research_transfer_neural_training as training_fixture
from tests import test_research_transfer_neural_recovery as recovery_fixture


class SharedPredictiveReplayTests(unittest.TestCase):
    def setUp(self):
        fixture = training_fixture.SharedNeuralTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.root, self.cells, self.plan = fixture.root, fixture.cells, fixture.plan
        # A badly predicted normal target remains in both metric and replay grids.
        self.cells["t2"]["soh"][:] = .01
        self.bundle, _ = train_shared_source_unit(self.cells, self.plan, [20, 50, 100], "gru", fixture.params, 9,
            "cpu", fixture.config, self.root / "source_refit", fixed_refit_epochs=1)
        self.metrics, self.arrays = evaluate_transfer(self.bundle, self.cells, self.plan["target_test"], predictor=predict_transfer_sequence)

    def test_actual_saved_cpu_model_replays_every_target_origin_and_nonvacuous_n100(self):
        with patch("torch.optim.AdamW", side_effect=AssertionError("replay fitted")):
            restored, _ = load_shared_neural_checkpoint(self.root / "source_refit")
            rows, arrays = replay.replay_unit(restored, self.cells, self.plan, self.arrays)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["passed"] and r["future_input_rows_deleted"] and r["poisoned_future_row_appended"] for r in rows))
        self.assertGreater(self.metrics["summary"]["100"]["cell_macro_mape_percent"], 6)
        self.assertEqual({r["cell_id"] for r in rows}, {"t1", "t2"})
        for key, value in arrays.items():
            if key.endswith("__intact"):
                prefix = key[:-len("intact")]
                np.testing.assert_array_equal(value, arrays[prefix+"prefix_only"])
                np.testing.assert_array_equal(value, arrays[prefix+"poisoned_future"])
                np.testing.assert_array_equal(value, self.arrays[prefix+"prediction"])

    def test_deleted_prefix_does_not_mutate_original_cells_and_removes_labels(self):
        before = copy.deepcopy(self.cells["t1"])
        visible = replay.prefix_only(self.cells["t1"], 20)
        self.assertNotIn("soh", visible)
        self.assertNotIn("label_cycle", visible)
        self.assertLessEqual(visible["cycle"].max(), 20)
        for name in ("cycle", "common", "label_cycle", "soh"):
            np.testing.assert_array_equal(self.cells["t1"][name], before[name])

    def test_changed_saved_predictions_fail_and_keep_failed_progress_without_refitting(self):
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        arrays["t1__20__prediction"] += .01
        progress = []
        with self.assertRaisesRegex(ValueError, "failed saved-prediction"):
            replay.replay_unit(self.bundle, self.cells, self.plan, arrays,
                progress_callback=lambda rows, outputs: progress.extend(copy.deepcopy(rows[-1:])))
        self.assertFalse(progress[-1]["saved_prediction_match"])
        self.assertTrue(progress[-1]["same_device_no_future_invariance"])
        self.assertFalse(progress[-1]["passed"])
        self.assertEqual(progress[-1]["atol"], 1e-5)

    def test_future_or_label_dependent_predictor_rejected_including_at_n100(self):
        original = replay.predict_transfer_sequence
        for kind in ("label", "length", "future"):
            def leaking(bundle, cell, origin, future, **kwargs):
                result = original(bundle, cell, origin, future, **kwargs)
                if kind == "label":
                    return result + (.01 if "soh" in cell else 0)
                if kind == "length":
                    return result + len(cell["cycle"])*.001
                return result + (.01 if np.isinf(cell["common"]).any() else 0)
            with self.subTest(kind=kind), patch.object(replay, "predict_transfer_sequence", side_effect=leaking):
                with self.assertRaisesRegex(ValueError, "causal replay"):
                    replay.replay_unit(self.bundle, self.cells, self.plan, self.arrays, origins=[100])

    def test_valid_coverage_gaps_are_reported_not_hidden_and_empty_replay_rejected(self):
        self.cells["t2"]["reference_established_source_cycle"] = 65
        report, arrays = evaluate_transfer(self.bundle, self.cells, self.plan["target_test"], predictor=predict_transfer_sequence)
        rows, _ = replay.replay_unit(self.bundle, self.cells, self.plan, arrays)
        self.assertEqual(sum(r.get("passed", False) for r in rows), 4)
        self.assertEqual(sum("status" in r for r in rows), 2)
        self.assertEqual(len(report["skipped"]), 2)
        only_target2 = {**self.plan, "target_test": ["t2"]}
        with self.assertRaisesRegex(ValueError, "no originally evaluated"):
            replay.replay_unit(self.bundle, self.cells, only_target2, arrays, origins=[20, 50])

    def test_invalid_origin_limit_grid_and_prediction_shape_rejected(self):
        for kwargs in ({"origins": []}, {"origins": [100, 100]}, {"origins": [101]}, {"max_cells": 0}, {"max_cells": True}):
            with self.assertRaises(ValueError):
                replay.replay_unit(self.bundle, self.cells, self.plan, self.arrays, **kwargs)
        changed = {key: value.copy() for key, value in self.arrays.items()}
        changed["t1__20__cycle"][0] = 20
        with self.assertRaisesRegex(ValueError, "invalid saved"):
            replay.replay_unit(self.bundle, self.cells, self.plan, changed)
        with patch.object(replay, "predict_transfer_sequence", return_value=np.array([.8])):
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                replay.replay_unit(self.bundle, self.cells, self.plan, self.arrays)

    def test_fresh_interpreter_replays_actual_checkpoint_with_real_frozen_model_resolution(self):
        from scripts.train_soh_research_transfer_neural_v5 import SOURCES
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
            descriptors[key] = {field: value for field, value in cell.items() if not isinstance(value, np.ndarray)}
            arrays.update({key+"__"+field: value for field, value in cell.items() if isinstance(value, np.ndarray)})
        save_json(self.root / "synthetic_cells.json", descriptors)
        np.savez_compressed(self.root / "synthetic_cells.npz", **arrays)
        np.savez_compressed(self.root / "saved_predictions.npz", **self.arrays)
        code = """
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
sys.path.insert(0,str(run/'source_snapshot'))
import battery_smetimes,scripts
battery_smetimes.__path__=[str(run/'source_snapshot/battery_smetimes'),*battery_smetimes.__path__]
scripts.__path__=[str(run/'source_snapshot/scripts'),*scripts.__path__]
from battery_smetimes.research_transfer_neural_replay import verify_model_resolution,replay_unit
from battery_smetimes.research_transfer_neural_training import load_shared_neural_checkpoint
from battery_smetimes.research_transfer_neural_recovery import verify_root_utility_resolution
import torch,numpy as np
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
def no_fit(*args,**kwargs):
 raise AssertionError('replay attempted training')
torch.optim.AdamW=no_fit
config=json.loads((run/'config.json').read_text())
paths=verify_model_resolution(run,config)
root_paths=verify_root_utility_resolution(run,config)
cells=json.loads((run/'synthetic_cells.json').read_text())
with np.load(run/'synthetic_cells.npz',allow_pickle=False) as arrays:
 for name in arrays.files:
  key,field=name.split('__')
  cells[key][field]=arrays[name]
bundle,evidence=load_shared_neural_checkpoint(run/'source_refit')
with np.load(run/'saved_predictions.npz',allow_pickle=False) as saved:
 rows,_=replay_unit(bundle,cells,config['plan'],saved)
print(json.dumps({'checks':len(rows),'passed':all(row['passed'] for row in rows),'source_paths':paths,'root_paths':root_paths,'no_fit':evidence['no_fit']}))
"""
        result = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-W", "error::UserWarning",
            "-c", code, str(self.root)], cwd=workspace, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout)
        self.assertEqual(evidence["checks"], 6)
        self.assertTrue(evidence["passed"] and evidence["no_fit"])
        self.assertTrue(all(Path(path).is_relative_to(self.root / "source_snapshot") for path in evidence["source_paths"].values()))
        self.assertEqual(len(evidence["root_paths"]), 5)
        self.assertTrue(all(Path(path).is_relative_to(self.root / "source_snapshot") for path in evidence["root_paths"].values()))


class SharedReplayParentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "shared_runs/run"
        source = self.run / "source_snapshot/toy_source.py"
        source.parent.mkdir(parents=True)
        source.write_text("# toy contract only, not a model implementation\n")
        prepared = self.root / "prepared"
        prepared.mkdir()
        for name in ("config.json", "manifest.json", "splits.json"):
            save_json(prepared / name, {"toy": True})
        self.config = {"prepare_only": False, "source_sha256": {"toy_source.py": stream_sha(source)},
            "packages": {"numpy": importlib.metadata.version("numpy")}, "prepared": str(prepared),
            "prepared_config_sha256": stream_sha(prepared / "config.json"),
            "prepared_manifest_sha256": stream_sha(prepared / "manifest.json"),
            "prepared_splits_sha256": stream_sha(prepared / "splits.json"), "main_run_bindings": []}
        self.state = {"complete": True, "model_training_started": True}
        save_json(self.run / "config.json", self.config)
        save_json(self.run / "progress.json", self.state)
        save_json(self.run / "splits.json", {"directions": []})
        save_json(self.run / "static_run_audit.json", {"finalized_units": 40, "toy": True})

    def test_prepare_partial_failed_source_or_input_change_rejected_before_raw_or_weights(self):
        original = (self.run / "config.json").read_bytes()
        for kind in ("prepare", "partial", "not_started", "failure", "source", "input", "package"):
            with self.subTest(kind=kind):
                config, state = copy.deepcopy(self.config), copy.deepcopy(self.state)
                if kind == "prepare":
                    config["prepare_only"] = True
                elif kind == "partial":
                    state["complete"] = False
                elif kind == "not_started":
                    state["model_training_started"] = False
                elif kind == "failure":
                    save_json(self.run / "failure.json", {"injected": True})
                elif kind == "source":
                    config["source_sha256"]["toy_source.py"] = "wrong"
                elif kind == "input":
                    config["prepared_manifest_sha256"] = "wrong"
                else:
                    config["packages"]["numpy"] = "wrong"
                save_json(self.run / "config.json", config)
                save_json(self.run / "progress.json", state)
                with patch.object(replay, "load_shared_cells", side_effect=AssertionError("raw data loaded too early")), \
                     patch("joblib.load", side_effect=AssertionError("weights loaded too early")):
                    with self.assertRaises(ValueError):
                        replay.verify_replay_run(self.run)
                if (self.run / "failure.json").exists():
                    (self.run / "failure.json").unlink()  # Only this explicit temporary injected test artifact.
        (self.run / "config.json").write_bytes(original)

    def test_original_main_gates_and_static_run_evidence_consumed_without_weight_loading(self):
        with patch.object(replay, "assert_main_complete", return_value={"toy_main": True}), \
             patch.object(replay, "assert_main_numerical_audits", return_value={"toy_numeric": True}), \
             patch.object(replay, "verify_root_utility_resolution", return_value={"toy_root_utilities": True}), \
             patch.object(replay, "load_shared_cells", return_value={"toy_cell": {}}), \
             patch.object(replay, "audit_shared_neural_run", return_value={"finalized_units": 40, "toy": True}), \
             patch("joblib.load", side_effect=AssertionError("verification must not deserialize")):
            verified = replay.verify_replay_run(self.run)
        self.assertFalse(verified["evidence"]["weights_loaded"])
        self.assertEqual(verified["evidence"]["main_completion"], {"toy_main": True})
        self.assertEqual(verified["evidence"]["static_finalized_units"], 40)

    def test_current_workspace_prediction_classes_cannot_pass_frozen_source_guard(self):
        with self.assertRaisesRegex(ValueError, "outside frozen source"):
            replay.verify_model_resolution(self.run, self.config)

    def test_actual_unit_parent_and_weight_checks_precede_no_fit_reload(self):
        fixture = recovery_fixture.SharedRecoveryTests()
        fixture.setUp()
        # fixture registers cleanup for its temporary directory; transfer it.
        self.addCleanup(fixture.doCleanups)
        fixture.execute(resume=False, session="synthetic")
        run = fixture.root
        config = {**fixture.config, "source_sha256": {}, "seeds": [9], "models": ["gru"]}
        save_json(run / "config.json", config)
        save_json(run / "progress.json", {"toy_finalized_grid": True})
        save_json(run / "splits.json", {"directions": [fixture.plan]})
        save_json(run / "static_run_audit.json", {"toy_parent": True})
        evidence = {key: stream_sha(run / name) for name, key in (("config.json", "run_config_sha256"),
            ("progress.json", "progress_sha256"), ("splits.json", "splits_sha256"), ("static_run_audit.json", "static_run_audit_sha256"))}
        verified = {"run": run, "config": config, "plans": [fixture.plan], "cells": fixture.cells, "evidence": evidence}
        # These tests isolate actual unit provenance/loading; they deliberately
        # do NOT claim a verified four-model public full-run contract.
        with patch.object(replay, "verify_model_resolution", return_value={"toy_resolution": True}), \
             patch("torch.optim.AdamW", side_effect=AssertionError("reload fitted")):
            bundle, loaded = replay.load_verified_unit(verified, fixture.plan, 9, "gru")
        self.assertTrue(loaded["checkpoint"]["no_fit"])
        self.assertEqual(loaded["device"], "cpu")
        self.assertEqual(bundle["source_train_ids"], fixture.plan["source_refit"])
        for name in ("config.json", "progress.json", "splits.json", "static_run_audit.json"):
            path, original = run / name, (run / name).read_bytes()
            path.write_bytes(b"changed")
            try:
                with patch("joblib.load", side_effect=AssertionError("changed parent deserialized")):
                    with self.assertRaisesRegex(ValueError, "parent evidence changed"):
                        replay.load_verified_unit(verified, fixture.plan, 9, "gru")
            finally:
                path.write_bytes(original)
        path = fixture.unit / "source_refit/model.pt"
        path.write_bytes(b"changed")
        with patch.object(replay, "verify_model_resolution", return_value={"toy_resolution": True}), \
             patch("joblib.load", side_effect=AssertionError("changed weights deserialized")):
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                replay.load_verified_unit(verified, fixture.plan, 9, "gru")


if __name__ == "__main__":
    unittest.main()
