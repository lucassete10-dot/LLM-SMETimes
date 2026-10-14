from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.benchmark_soh_research_v5 import (enforce_idle, latency_summary, model_description,
    timed_prediction, validate_query_equivalence, visible_input)
from scripts.prepare_same_type_forecast import save_json
from scripts.report_soh_research_resources_v5 import collect


class ResourceMeasurementTests(unittest.TestCase):
    def test_technical_repeat_spread_is_sample_sd_not_extra_batteries(self):
        result = latency_summary([1, 2, 3])
        self.assertEqual(result["technical_repeats"], 3)
        self.assertEqual(result["mean_ms"], 2)
        self.assertEqual(result["sd_ms"], 1)
        self.assertIsNone(latency_summary([1])["sd_ms"])
        for values in ([], [np.nan], [-1]):
            with self.assertRaises(ValueError):
                latency_summary(values)

    def test_visible_input_contains_no_labels_or_future_rows_and_does_not_mutate(self):
        original = {"cycle": np.arange(1, 6), "raw": np.arange(10).reshape(5, 2), "soh": np.ones(5),
            "label_cycle": np.arange(1, 6), "unfiltered_soh": np.ones(5),
            "physical_family": "family", "reference_established_source_cycle": 1, "metadata": {"dataset": "MIT"}}
        value = visible_input(original, 3)
        self.assertEqual(set(value), {"cycle", "raw", "metadata", "physical_family", "reference_established_source_cycle"})
        np.testing.assert_array_equal(value["cycle"], [1, 2, 3])
        value["raw"][:] = -1
        value["metadata"]["dataset"] = "NASA"
        self.assertEqual(original["metadata"]["dataset"], "MIT")
        self.assertEqual(original["raw"][0, 0], 0)

    def test_idle_guard_does_not_promote_contended_runs_to_formal_comparison(self):
        empty_gpu = {"auditable": True, "workers": []}
        with self.assertRaisesRegex(ValueError, "idle preflight failed"):
            enforce_idle([{"pid": 1}], empty_gpu, "cpu", False)
        self.assertTrue(enforce_idle([{"pid": 1}], empty_gpu, "cpu", True))
        self.assertFalse(enforce_idle([], empty_gpu, "cuda", False))
        with self.assertRaises(ValueError):
            enforce_idle([], {"auditable": False, "workers": []}, "cuda", False)
        self.assertTrue(enforce_idle([], {"auditable": False, "workers": []}, "cuda", True))

    def test_total_trainable_and_frozen_parameter_counts_are_distinct(self):
        network = torch.nn.Linear(3, 1)
        network.bias.requires_grad = False
        result = model_description({"network": network})
        self.assertEqual(result["total_parameters"], 4)
        self.assertEqual(result["trainable_parameters"], 3)
        self.assertEqual(result["frozen_parameters"], 1)
        self.assertEqual(result["parameter_storage_bytes"], 16)

    def test_tree_nodes_are_not_mislabeled_as_neural_parameters(self):
        model = SimpleNamespace(estimators_=[SimpleNamespace(tree_=SimpleNamespace(node_count=3))])
        result = model_description({"model": model})
        self.assertIsNone(result["total_parameters"])
        self.assertEqual(result["tree_nodes_not_neural_parameters"], 3)

    def test_cuda_timing_synchronizes_and_peak_is_process_allocator(self):
        events = []
        cuda = SimpleNamespace(synchronize=lambda device: events.append("sync"),
            memory_allocated=lambda device: 30, reset_peak_memory_stats=lambda device: events.append("reset"),
            max_memory_allocated=lambda device: 40, max_memory_reserved=lambda device: 80)
        bundle = {"card_cache": {"old": "embedding"}}
        def predictor(bundle, cell, origin, queries):
            self.assertNotIn("card_cache", bundle)
            events.append("predict")
            return np.array([.8])
        with patch("scripts.benchmark_soh_research_v5.process_memory", return_value={}):
            output, elapsed, memory = timed_prediction(predictor, bundle, {}, 100, [101], "cuda", True, SimpleNamespace(cuda=cuda))
        self.assertEqual(events, ["sync", "reset", "predict", "sync"])
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(memory["cuda_incremental_peak_allocated_bytes"], 10)
        self.assertEqual(memory["cuda_peak_reserved_bytes"], 80)

    def test_cache_speedup_cannot_change_output_and_query_count_cannot_condition_soh(self):
        value = np.array([.8, .7])
        self.assertEqual(validate_query_equivalence(value, value.copy(), .8, True), 0)
        with self.assertRaisesRegex(ValueError, "cache modes"):
            validate_query_equivalence(value, value + .001, .8, True)
        with self.assertRaisesRegex(ValueError, "curve length"):
            validate_query_equivalence(value, None, .81, True)

    def test_tabular_arithmetic_drift_is_bounded_without_claiming_bitwise_identity(self):
        value = np.array([.8, .7])
        self.assertEqual(validate_query_equivalence(value, value + 1e-16, .8, False), 0)
        with self.assertRaisesRegex(ValueError, "cache modes"):
            validate_query_equivalence(value, value + 1e-6, .8, False)
        with self.assertRaisesRegex(ValueError, "cache modes"):
            validate_query_equivalence(value, value + 1e-8, .8, True)

    def test_failed_resource_attempt_with_utf8_metadata_is_retained(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            save_json(folder / "config.json", {"processor": "测试处理器"})
            save_json(folder / "failure.json", {"traceback": "空闲检查未通过"})
            run, cases = collect(folder)
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["failure"]["traceback"], "空闲检查未通过")
            self.assertEqual(cases, [])


if __name__ == "__main__":
    unittest.main()
