import unittest
from types import SimpleNamespace

import numpy as np
import torch

from battery_smetimes.data.battery_dataset import fit_source_imputer
from battery_smetimes.data.cycle_time import (
    CycleTimeWindowDataset, causal_cycle_slopes,
)
from battery_smetimes.models.cycle_time import CycleTimeHealthModel
from battery_smetimes.models.temporal import BatteryHealthGRU, BatteryHealthTCN


def fixture(cycle, values):
    raw = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return SimpleNamespace(
        raw=raw, cycle=np.asarray(cycle), dataset="XJTU",
        matrix=lambda variant: np.tile(raw, (1, 3)),
        soh=np.ones(len(raw)), soh_available=np.ones(len(raw), dtype=bool),
        capacity_review_flag=np.zeros(len(raw), dtype=bool))


class CycleTimeTest(unittest.TestCase):
    def test_slope_uses_real_cycles_and_is_causal(self):
        cycle = np.array([1, 6, 11, 16, 21, 26])
        signal = (2 * cycle + 7)[:, None].astype(float)
        slopes = causal_cycle_slopes(cycle, signal)
        np.testing.assert_allclose(slopes[1:], 2)
        future_changed = signal.copy()
        future_changed[4:] += 1000
        np.testing.assert_array_equal(
            causal_cycle_slopes(cycle, future_changed)[:4], slopes[:4])
        with self.assertRaises(ValueError):
            causal_cycle_slopes([1, 1], [[1], [2]])

    def test_context_has_real_elapsed_time_without_cell_or_future_leakage(self):
        cells = {"a": fixture([1, 6, 11, 16], [1, 2, 3, 4]),
                 "b": fixture([1, 2, 3], [100, 200, 300])}
        imputer = fit_source_imputer(cells, ["a"])
        common = dict(cells=cells, records_by_id={"a": {}, "b": {}},
                      cell_ids=["a", "b"], imputer=imputer, lookback=3)
        baseline = CycleTimeWindowDataset(**common, time_mode="baseline")
        elapsed = CycleTimeWindowDataset(**common, time_mode="elapsed")
        rate = CycleTimeWindowDataset(**common, time_mode="elapsed_rate")
        self.assertEqual(baseline.context_width, rate.context_width)
        self.assertTrue(torch.all(baseline[2]["cycle_context"] == 0))
        np.testing.assert_allclose(elapsed[2]["cycle_context"][:, 0], [-.1, -.05, 0])
        np.testing.assert_allclose(elapsed[2]["cycle_context"][:, 1], [0, .5, .5])
        self.assertTrue(torch.all(elapsed[2]["cycle_context"][:, 2:] == 0))
        self.assertTrue(torch.all(rate[4]["cycle_context"] == 0))
        prefix = rate[2]["cycle_context"].clone()
        cells["a"] = fixture([1, 6, 11, 900], [1, 2, 3, 999])
        changed = CycleTimeWindowDataset(**common, time_mode="elapsed_rate")
        torch.testing.assert_close(prefix, changed[2]["cycle_context"])

    def test_zero_context_exactly_reproduces_base_and_real_context_is_trainable(self):
        for name, factory in (("gru", BatteryHealthGRU), ("tcn", BatteryHealthTCN)):
            torch.manual_seed(7)
            original = factory(3, physics_future=True).eval()
            torch.manual_seed(7)
            timed = CycleTimeHealthModel(3, context_width=3, backbone=name).eval()
            for key, tensor in original.state_dict().items():
                torch.testing.assert_close(tensor, timed.core.state_dict()[key])
            signal, metadata = torch.randn(2, 4, 3), torch.randn(2, 6)
            mask = torch.ones(2, 4, dtype=torch.bool)
            context = torch.randn(2, 4, 3)
            legacy = original(signal, mask, metadata, (10, 50, 100))
            adapted = timed(signal, mask, metadata, (10, 50, 100),
                            cycle_context=context)
            torch.testing.assert_close(legacy["future_soh"], adapted["future_soh"])
            adapted["future_soh"].sum().backward()
            self.assertGreater(float(timed.context_projection.weight.grad.abs().sum()), 0)
            with torch.no_grad():
                timed.context_projection.weight.normal_(std=.2)
            changed = timed(signal, mask, metadata, (10, 50, 100),
                            cycle_context=context)
            self.assertFalse(torch.allclose(changed["future_soh"], legacy["future_soh"]))
            zero = timed(signal, mask, metadata, (10, 50, 100),
                         cycle_context=torch.zeros_like(context))
            torch.testing.assert_close(zero["future_soh"], legacy["future_soh"])


if __name__ == "__main__":
    unittest.main()
