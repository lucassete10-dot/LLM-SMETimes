import unittest
from types import SimpleNamespace

import numpy as np

from scripts.replay_soh_research_v5 import causal_pair, replay_comparison, resolved_device


class ReplayEvidenceTests(unittest.TestCase):
    def test_default_reuses_original_device_without_silent_cpu_fallback(self):
        self.assertEqual(resolved_device("auto", {"device": "cuda"}), "cuda")
        self.assertEqual(resolved_device("auto", {}), "cpu")
        self.assertEqual(resolved_device("cpu", {"device": "cuda"}), "cpu")

    def test_same_device_invariance_and_saved_match_are_separate(self):
        visible = np.array([.8, .7])
        result = replay_comparison(visible, visible.copy(), visible + 4e-5, True)
        self.assertTrue(result["same_device_no_future_invariance"])
        self.assertFalse(result["saved_prediction_match"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["tolerance"], 1e-5)

    def test_small_causal_change_is_not_hidden_by_saved_roundoff_tolerance(self):
        visible = np.array([.8, .7])
        result = replay_comparison(visible, visible + 1e-8, visible, True)
        self.assertTrue(result["saved_prediction_match"])
        self.assertFalse(result["same_device_no_future_invariance"])
        self.assertFalse(result["passed"])

    def test_exact_replay_passes_both_checks(self):
        visible = np.array([.8, .7])
        result = replay_comparison(visible, visible.copy(), visible.copy(), False)
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_absolute_soh_difference"], 0)

    def test_serial_forest_control_restores_original_runtime_and_preserves_primary_tolerance(self):
        bundle = {"model": SimpleNamespace(estimators_=[1], n_jobs=4)}
        def predictor(bundle, cell, origin, future):
            self.assertEqual(bundle["model"].n_jobs, 1)
            return np.array([.8])
        original_runtime_replay = np.array([.8 + 1e-16])
        changed, intact, control = causal_pair(predictor, bundle, {}, {}, 100, [101], original_runtime_replay)
        self.assertEqual(bundle["model"].n_jobs, 4)
        self.assertTrue(control["serial_tree_accumulation_control"])
        result = replay_comparison(original_runtime_replay, intact, np.array([.8]), False, changed)
        self.assertTrue(result["passed"])
        self.assertGreater(result["max_absolute_soh_difference"], 0)
        self.assertEqual(result["same_device_invariance_max_absolute_difference"], 0)

    def test_failed_serial_control_cannot_leave_the_model_runtime_modified(self):
        bundle = {"model": SimpleNamespace(estimators_=[1], n_jobs=4)}
        def predictor(*args):
            raise ValueError("test exception")
        with self.assertRaisesRegex(ValueError, "test exception"):
            causal_pair(predictor, bundle, {}, {}, 100, [101], np.array([.8]))
        self.assertEqual(bundle["model"].n_jobs, 4)


if __name__ == "__main__":
    unittest.main()
