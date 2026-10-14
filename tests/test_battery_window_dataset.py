import unittest
from collections import Counter
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from battery_smetimes.data.battery_dataset import (
    CausalWindowDataset, DomainCellBalancedSampler, fit_source_imputer,
    fit_source_scaler, WaveformWindowDataset,
)
from battery_smetimes.data.split import cellwise_split


def cell(dataset, cycle, values, soh):
    features = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return SimpleNamespace(
        dataset=dataset, cycle=np.asarray(cycle),
        soh=np.asarray(soh, dtype=np.float32),
        soh_available=np.isfinite(soh), matrix=lambda variant: features,
        capacity_review_flag=np.zeros(len(cycle), dtype=bool),
    )


class BatteryWindowDatasetTest(unittest.TestCase):
    def setUp(self):
        self.cells = {
            "a": cell("NASA", [1, 2, 3], [1, 2, 3], [np.nan, .9, .8]),
            "b": cell("MIT", [7, 8], [100, 200], [.95, .85]),
        }
        self.records = {
            "a": {"ambient_temperature_c": None, "charge_rate_c": None,
                  "discharge_rate_c": 1.0},
            "b": {"ambient_temperature_c": 30, "charge_rate_c": 1.0,
                  "discharge_rate_c": 4.0},
        }

    def test_windows_never_cross_cell_or_include_future(self):
        imputer = fit_source_imputer(self.cells, ["a"])
        dataset = CausalWindowDataset(self.cells, self.records, ["a", "b"],
                                      imputer, lookback=3)
        self.assertEqual(len(dataset), 4)
        first_b = dataset[2]
        self.assertEqual(first_b["cell_id"], "b")
        self.assertEqual(first_b["target_cycle"], 7)
        self.assertEqual(first_b["mask"].tolist(), [False, False, True])
        self.assertEqual(first_b["signal"][:, 0].tolist(), [0.0, 0.0, 100.0])
        self.assertEqual(dataset[3]["relative_cycle"].tolist(), [0.0, -1.0, 0.0])
        self.assertEqual(dataset[1]["signal"][-1, 0].item(), 3.0)

    def test_imputer_is_fitted_on_train_cells_only(self):
        self.cells["b"] = cell("MIT", [7, 8], [np.nan, np.nan], [.95, .85])
        imputer = fit_source_imputer(self.cells, ["a"])
        dataset = CausalWindowDataset(self.cells, self.records, ["b"],
                                      imputer, lookback=2)
        self.assertEqual(dataset[0]["signal"][-1, 0].item(), 2.0)

    def test_scaler_and_future_targets_are_causal_inputs_only(self):
        imputer = fit_source_imputer(self.cells, ["a"])
        scaler = fit_source_scaler(self.cells, ["a"], imputer)
        dataset = CausalWindowDataset(self.cells, self.records, ["a"],
                                      imputer, scaler=scaler, lookback=2,
                                      future_horizons=(1, 2))
        first = dataset[0]
        self.assertEqual(first["target_cycle"], 2)
        self.assertEqual(first["future_available"].tolist(), [True, False])
        self.assertAlmostEqual(first["future_soh"][0].item(), .8, places=5)
        self.assertEqual(first["signal"].shape[0], 2)

    def test_domain_cell_balanced_sampling(self):
        imputer = fit_source_imputer(self.cells, ["a"])
        dataset = CausalWindowDataset(self.cells, self.records, ["a", "b"],
                                      imputer, lookback=2)
        sampler = DomainCellBalancedSampler(dataset, 1000, seed=17)
        domains = Counter(dataset.rows[index][0] for index in sampler)
        self.assertLess(abs(domains["a"] - domains["b"]), 100)
        sampler.set_epoch(1)
        self.assertNotEqual(list(DomainCellBalancedSampler(dataset, 10, seed=17)),
                            list(DomainCellBalancedSampler(dataset, 10, seed=18)))

    def test_waveform_window_stays_inside_cell_and_aligns_cycles(self):
        imputer = fit_source_imputer(self.cells, ["a"])
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "a.npz"
            wave = np.zeros((3, 2, 4, 121), dtype=np.float32)
            wave[:, :, 0, :] = np.arange(1, 4)[:, None, None]
            wave[:, 1, 0, :] += 100
            np.savez_compressed(source, cycle=[1, 2, 3], waveform=wave)
            dataset = WaveformWindowDataset(self.cells, self.records, ["a"],
                                            imputer, waveform_dir=temp, lookback=3)
            self.assertEqual(tuple(dataset[0]["waveform"].shape), (3, 2, 4, 121))
            self.assertEqual(dataset[0]["waveform"][:, 0, 0, 0].tolist(),
                             [0.0, 1.0, 2.0])
            charge_only = WaveformWindowDataset(
                self.cells, self.records, ["a"], imputer, waveform_dir=temp,
                lookback=3, waveform_stage="charge")
            self.assertEqual(tuple(charge_only[0]["waveform"].shape),
                             (3, 1, 4, 121))
            self.assertEqual(charge_only[0]["waveform"][-1, 0, 0, 0].item(), 2.)
            np.savez_compressed(source, cycle=[1, 2, 4], waveform=wave)
            with self.assertRaisesRegex(ValueError, "cycle mismatch"):
                WaveformWindowDataset(self.cells, self.records, ["a"], imputer,
                                      waveform_dir=temp)

    def test_stratified_cell_split_is_disjoint_and_deterministic(self):
        records = [{"dataset": "XJTU", "cell_id": f"g{group}_c{index}",
                    "group": f"g{group}"}
                   for group in range(5) for index in range(11)]
        first = cellwise_split(records, "XJTU", seed=42)
        second = cellwise_split(records, "XJTU", seed=42)
        self.assertEqual(first, second)
        self.assertEqual(sum(len(v) for v in first.values()), 55)
        self.assertEqual(len(set().union(*map(set, first.values()))), 55)
        self.assertTrue(all(first.values()))


if __name__ == "__main__":
    unittest.main()
