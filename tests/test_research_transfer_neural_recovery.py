"""Actual synthetic CPU recovery/failure injection, not public transfer scores."""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from battery_smetimes.research_transfer_neural_recovery import (
    archive_interrupted, exclusive_run_lock, recover_source_stage, execute_shared_unit,
)
from battery_smetimes.research_transfer_neural_training import train_shared_source_unit, _save_scores
from battery_smetimes.research_transfer_sequences import predict_transfer_sequence
from battery_smetimes.research_transfer_signals import RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.prepare_same_type_forecast import save_json
from tests import test_research_transfer_neural_training as training_fixture


class SharedRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = training_fixture.SharedNeuralTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.root, self.cells, self.plan, self.fold, self.params = fixture.root, fixture.cells, fixture.plan, fixture.fold, fixture.params
        self.settings = {**fixture.config, "candidates": [self.params], "smoke_epochs": 2, "patience_validations": 3}
        self.source = {"configuration": self.settings, "checkpoint_files_sha256": {}, "binding": {"run": "toy"}}
        self.config = {"smoke": True, "origins": [20, 50, 100], "device": "cpu", "model_sources": {"gru": self.source}}
        self.unit = self.root / self.plan["name"] / "9" / "gru"
        self.stage = self.unit / "candidate_0" / self.fold["name"]

    def stage_recovery(self, *, resume=True, session="recovery", destination=None, fold=None, epochs=None, need_bundle=False):
        return recover_source_stage(self.cells, self.plan, [20, 50, 100], "gru", self.params, 9, "cpu", self.source,
            self.stage if destination is None else destination, run=self.root, session=session, resume=resume,
            fold=self.fold if fold is None and epochs is None else fold, epochs=epochs, need_bundle=need_bundle)

    def fit_candidate(self):
        return train_shared_source_unit(self.cells, self.plan, [20, 50, 100], "gru", self.params, 9, "cpu", self.settings,
            self.stage, source_fold=self.fold)

    @staticmethod
    def hashes(folder):
        return {str(p.relative_to(folder)): stream_sha(p) for p in folder.rglob("*") if p.is_file()}

    def execute(self, *, resume, session):
        # Actual predictions/metrics, but no manuscript plots in unit tests.
        with exclusive_run_lock(self.root), patch("scripts.train_soh_research_v5.save_scores", side_effect=_save_scores):
            return execute_shared_unit(self.root, self.config, self.plan, 9, "gru", self.cells,
                resume=resume, session=session)

    def test_complete_candidate_reused_without_training_or_deserialization(self):
        _, expected = self.fit_candidate()
        hashes = self.hashes(self.stage)
        with patch("battery_smetimes.research_transfer_neural_training.train_shared_source_unit", side_effect=AssertionError("refit")), \
             patch("joblib.load", side_effect=AssertionError("candidate reuse deserialized")):
            bundle, actual = self.stage_recovery()
        self.assertIsNone(bundle)
        self.assertEqual(actual, expected)
        self.assertEqual(self.hashes(self.stage), hashes)
        events = json.loads((self.root / "resumes/recovery/events.json").read_text())
        self.assertEqual(events[-1]["event"], "reuse_audited_source_stage")

    def test_partial_second_epoch_archived_and_original_seed_restarted(self):
        from battery_smetimes import research_transfer_neural_training as training
        original, calls = training.fit_epoch, []
        def interrupted(*args, **kwargs):
            calls.append(args[4])
            if len(calls) == 2:
                raise RuntimeError("injected epoch interruption")
            return original(*args, **kwargs)
        with patch.object(training, "fit_epoch", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.fit_candidate()
        partial_hashes = self.hashes(self.stage)
        self.assertEqual(calls, [1, 2])
        bundle, report = self.stage_recovery(need_bundle=True)
        archived = self.root / "interrupted_attempts/recovery" / self.stage.relative_to(self.root)
        self.assertEqual(self.hashes(archived), partial_hashes)
        self.assertTrue(report["complete"])
        reference = self.root / "reference"
        expected, _ = training.train_shared_source_unit(self.cells, self.plan, [20, 50, 100], "gru", self.params, 9,
            "cpu", self.settings, reference, source_fold=self.fold)
        actual = predict_transfer_sequence(bundle, self.cells["t1"], 100, [110, 120], mode=RESEARCH_MODE)
        np.testing.assert_array_equal(actual, predict_transfer_sequence(expected, self.cells["t1"], 100, [110, 120], mode=RESEARCH_MODE))
        for key, tensor in bundle["network"].state_dict().items():
            self.assertTrue(torch.equal(tensor, expected["network"].state_dict()[key]))
        self.assertEqual(json.loads((self.stage / "config.json").read_text())["seed"], 9)

    def test_changed_complete_checkpoint_or_partial_configuration_never_retrained(self):
        self.fit_candidate()
        path = self.stage / "model.pt"
        path.write_bytes(b"tampered")
        with patch("battery_smetimes.research_transfer_neural_training.train_shared_source_unit", side_effect=AssertionError("refit")):
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                self.stage_recovery()
        self.assertFalse((self.root / "interrupted_attempts").exists())
        config = json.loads((self.stage / "config.json").read_text())
        for key, value in (("seed", 10), ("device", "cuda"), ("parameters", {}), ("source_fold", {})):
            changed = {**config, key: value}
            save_json(self.stage / "config.json", changed)
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.stage_recovery()
        save_json(self.stage / "config.json", config)

    def test_missing_identity_or_implicit_resume_rejected_without_moving(self):
        self.stage.mkdir(parents=True)
        save_json(self.stage / "failure.json", {"injected": True})
        for resume in (False, True):
            with self.assertRaises(ValueError):
                self.stage_recovery(resume=resume)
        self.assertTrue((self.stage / "failure.json").is_file())
        self.assertFalse((self.root / "interrupted_attempts").exists())

    def test_refit_completed_before_target_crash_is_reloaded_not_retrained(self):
        self.unit.mkdir(parents=True)
        save_json(self.unit / "config.json", {"plan": self.plan, "seed": 9, "model": "gru", "source": self.source, "folds": [self.fold]})
        self.fit_candidate()
        from battery_smetimes import research_transfer_baselines as baseline
        original = baseline.evaluate_transfer
        def fail_target(bundle, cells, ids, **kwargs):
            if ids == self.plan["target_test"]:
                raise RuntimeError("target interruption")
            return original(bundle, cells, ids, **kwargs)
        with patch.object(baseline, "evaluate_transfer", side_effect=fail_target):
            with self.assertRaisesRegex(RuntimeError, "target interruption"):
                self.execute(resume=True, session="first")
        candidate_hashes, refit_hashes = self.hashes(self.stage), self.hashes(self.unit / "source_refit")
        with patch("battery_smetimes.research_transfer_neural_training.train_shared_source_unit", side_effect=AssertionError("refit")):
            result = self.execute(resume=True, session="second")
        self.assertEqual(self.hashes(self.stage), candidate_hashes)
        self.assertEqual(self.hashes(self.unit / "source_refit"), refit_hashes)
        self.assertEqual(result["target_cells"], 2)

    def test_partial_target_archived_without_retraining_source_or_changing_selection(self):
        def interrupted_save(folder, report, arrays):
            _save_scores(folder, report, arrays)
            raise RuntimeError("evaluation save interruption")
        with exclusive_run_lock(self.root), patch("scripts.train_soh_research_v5.save_scores", side_effect=interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "evaluation save interruption"):
                execute_shared_unit(self.root, self.config, self.plan, 9, "gru", self.cells, session="first", resume=False)
        partial = self.hashes(self.unit / "target_test")
        selected_digest = stream_sha(self.unit / "selected.json")
        with patch("battery_smetimes.research_transfer_neural_training.train_shared_source_unit", side_effect=AssertionError("refit")):
            self.execute(resume=True, session="second")
        archive = self.root / "interrupted_attempts/second" / (self.unit / "target_test").relative_to(self.root)
        self.assertEqual(self.hashes(archive), partial)
        self.assertEqual(stream_sha(self.unit / "selected.json"), selected_digest)

    def test_completed_unit_different_device_is_not_reinterpreted_as_same_run(self):
        self.execute(resume=False, session="first")
        self.config["device"] = "cuda"
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.execute(resume=True, session="second")

    def test_completed_unit_or_lost_progress_reused_without_prediction_or_fit(self):
        self.cells["t2"]["soh"][:] = .01  # Keep the difficult cell in recovery.
        expected = self.execute(resume=False, session="first")
        hashes = self.hashes(self.unit)
        with patch("battery_smetimes.research_transfer_neural_training.train_shared_source_unit", side_effect=AssertionError("refit")), \
             patch("battery_smetimes.research_transfer_baselines.evaluate_transfer", side_effect=AssertionError("predict")):
            result = self.execute(resume=True, session="second")
        self.assertEqual(result, expected)
        self.assertGreater(result["target_summary"]["100"]["cell_macro_mape_percent"], 6)
        self.assertEqual(self.hashes(self.unit), hashes)

    def test_completed_target_before_audit_crash_keeps_its_predictions(self):
        from battery_smetimes import research_transfer_neural_audit as audit
        with patch.object(audit, "audit_shared_neural_unit", side_effect=RuntimeError("audit interruption")):
            with self.assertRaisesRegex(RuntimeError, "audit interruption"):
                self.execute(resume=False, session="first")
        hashes = self.hashes(self.unit / "target_test")
        with patch("battery_smetimes.research_transfer_neural_training.train_shared_source_unit", side_effect=AssertionError("refit")), \
             patch("battery_smetimes.research_transfer_baselines.evaluate_transfer", side_effect=AssertionError("predict")), \
             patch("joblib.load", side_effect=AssertionError("unneeded deserialize")):
            self.execute(resume=True, session="second")
        self.assertEqual(self.hashes(self.unit / "target_test"), hashes)

    def test_bounded_archive_preserves_bytes_and_rejects_broad_or_colliding_paths(self):
        self.stage.mkdir(parents=True)
        save_json(self.stage / "failure.json", {"kept": True})
        for bad in (self.root, self.root.parent, self.unit, self.root / "source_snapshot", self.root / "resumes"):
            with self.assertRaises(ValueError):
                archive_interrupted(self.root, bad, "first")
        destination = archive_interrupted(self.root, self.stage, "first")
        self.assertEqual(json.loads((destination / "failure.json").read_text()), {"kept": True})
        self.stage.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            archive_interrupted(self.root, self.stage, "first")
        with self.assertRaises(ValueError):
            archive_interrupted(self.root, self.stage, "../escape")

    def test_real_os_lock_rejects_second_process_then_releases(self):
        code = "from pathlib import Path; from battery_smetimes.research_transfer_neural_recovery import exclusive_run_lock; " \
               "lock=exclusive_run_lock(Path(__import__('sys').argv[1])); lock.__enter__(); lock.__exit__(None,None,None)"
        workspace = Path(__file__).resolve().parents[1]
        with exclusive_run_lock(self.root):
            blocked = subprocess.run([sys.executable, "-c", code, str(self.root)], cwd=workspace, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("live OS-lock owner", blocked.stderr)
        released = subprocess.run([sys.executable, "-c", code, str(self.root)], cwd=workspace, capture_output=True, text=True, timeout=20)
        self.assertEqual(released.returncode, 0, released.stderr)

    def test_os_lock_releases_on_owner_process_termination_not_stale_file(self):
        code = "from pathlib import Path\nfrom battery_smetimes.research_transfer_neural_recovery import exclusive_run_lock\n" \
               "import sys,time\nwith exclusive_run_lock(Path(sys.argv[1])):\n print('locked',flush=True)\n time.sleep(30)\n"
        workspace = Path(__file__).resolve().parents[1]
        owner = subprocess.Popen([sys.executable, "-c", code, str(self.root)], cwd=workspace,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(owner.stdout.readline().strip(), "locked")
            with self.assertRaisesRegex(ValueError, "live OS-lock owner"):
                with exclusive_run_lock(self.root):
                    self.fail("concurrent worker acquired lock")
            owner.terminate()
            owner.wait(timeout=10)
            self.assertTrue((self.root / "worker.lock").is_file())
            with exclusive_run_lock(self.root):
                pass  # Lock file remains, but authoritative OS owner is gone.
        finally:
            if owner.poll() is None:
                owner.terminate()
                owner.wait(timeout=10)
            owner.stdout.close()
            owner.stderr.close()


if __name__ == "__main__":
    unittest.main()
