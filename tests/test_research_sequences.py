import copy
import unittest

import numpy as np
import torch

from battery_smetimes.physical_types import load_registry
from battery_smetimes.research_sequences import HistoryPreprocessor, SmallForecaster, predict_sequence


class ResearchSequenceTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()
        self.family = next(iter(self.registry["families"]))
        rng = np.random.default_rng(20)
        self.cells = {key: {"cycle": np.arange(1, 151), "raw": rng.normal(size=(150, 51)),
            "label_cycle": np.arange(1, 151), "soh": np.linspace(1, .8, 150),
            "reference_established_source_cycle": 1, "physical_family": self.family,
            "metadata": {"dataset": "MIT", "group": "batch1", "charge_protocol": "4C(80%)-4C",
                         "diagnostic_discharge_rate_c": 4}}
            for key in ("a", "b", "heldout")}
        torch.set_num_threads(2)

    def processor(self):
        return HistoryPreprocessor().fit(self.cells, ["a", "b"], [20, 50, 100])

    def test_history_scaling_ignores_unlisted_cells(self):
        before = self.processor()
        self.cells["heldout"]["raw"][:] = 1e6
        self.cells["heldout"]["soh"][:] = 1e7
        after = self.processor()
        np.testing.assert_array_equal(before.raw_scale.scaler.mean_, after.raw_scale.scaler.mean_)
        np.testing.assert_array_equal(before.stats_scale.imputer.statistics_, after.stats_scale.imputer.statistics_)
        self.assertEqual(set(before.train_ids), {"a", "b"})

    def test_input_does_not_require_capacity_soh_or_future_records(self):
        processor = self.processor()
        cell = self.cells["heldout"]
        before = processor.model_input(cell, 50)
        del cell["soh"], cell["label_cycle"]
        cell["raw"][50:] = np.inf
        after = processor.model_input(cell, 50)
        for field in ("history", "statistics", "conditions"):
            np.testing.assert_array_equal(before[field], after[field])

    def test_every_encoder_ignores_padding_after_true_length(self):
        processor = self.processor()
        values = processor.model_input(self.cells["heldout"], 20)
        history = torch.as_tensor(values["history"][None])
        changed = history.clone()
        changed[:, 20:] = 100
        statistics = torch.as_tensor(values["statistics"][None])
        length = torch.as_tensor([20])
        for architecture in ("gru", "lstm", "tcn", "transformer", "statistics_mlp"):
            model = SmallForecaster(architecture, 103, processor.stats_width, hidden=16, dropout=0).eval()
            with torch.inference_mode():
                np.testing.assert_allclose(model.encode(history, length, statistics).numpy(),
                    model.encode(changed, length, statistics).numpy(), rtol=1e-5, atol=1e-5)

    def test_horizon_queries_are_pointwise_and_cross_family_is_rejected(self):
        processor = self.processor()
        model = SmallForecaster("gru", 103, processor.stats_width, hidden=16, dropout=0).eval()
        bundle = {"network": model, "processor": processor, "device": "cpu", "physical_family": self.family,
                  "registry": self.registry}
        cell = self.cells["heldout"]
        prediction = predict_sequence(bundle, cell, 50, [60])
        np.testing.assert_allclose(prediction, predict_sequence(bundle, cell, 50, [60, 100, 10000])[:1], atol=1e-6)
        cell["physical_family"] = list(self.registry["families"])[2]
        with self.assertRaisesRegex(ValueError, "incompatible"):
            predict_sequence(bundle, cell, 50, [60])


if __name__ == "__main__":
    unittest.main()
