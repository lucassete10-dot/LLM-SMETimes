import copy
import unittest

import numpy as np

from battery_smetimes.physical_types import load_registry
from battery_smetimes.research_baselines import (fit_baseline, predict_baseline,
    regression_metrics, signal_prefix)


class ResearchBaselineTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()
        self.family = next(iter(self.registry["families"]))
        rng = np.random.default_rng(3)
        self.cells = {key: {"cycle": np.arange(1, 151), "raw": rng.normal(size=(150, 51)),
            "label_cycle": np.arange(1, 151), "soh": np.linspace(1, .8, 150),
            "reference_established_source_cycle": 1, "physical_family": self.family,
            "metadata": {"dataset": "MIT", "group": "batch1", "charge_protocol": "4C(80%)-4C",
                         "diagnostic_discharge_rate_c": 4}}
            for key in ("a", "b", "held_out")}
        self.config = {"origins_source_cycles": [20, 50, 100]}

    def fit(self):
        return fit_baseline(self.cells, ["a", "b"], self.family, self.registry,
                            self.config, "ridge", {"alpha": 10.0}, 123, smoke=True)

    def test_future_raw_and_soh_do_not_change_a_query(self):
        bundle = self.fit()
        cell = self.cells["held_out"]
        prediction = predict_baseline(bundle, cell, 50, [60, 100, 150])
        cell["raw"][50:] = np.inf
        cell["soh"][:] = 1000
        np.testing.assert_array_equal(prediction, predict_baseline(bundle, cell, 50, [60, 100, 150]))

    def test_held_out_data_cannot_fit_imputation_scaling_or_regression(self):
        before = self.fit()
        self.cells["held_out"]["raw"][:] = 1e7
        self.cells["held_out"]["soh"][:] = 1e5
        after = self.fit()
        np.testing.assert_array_equal(before["imputer"].statistics_, after["imputer"].statistics_)
        np.testing.assert_array_equal(before["scaler"].mean_, after["scaler"].mean_)
        np.testing.assert_array_equal(before["model"].coef_, after["model"].coef_)

    def test_batch_name_is_not_a_physical_type_feature(self):
        cell = self.cells["a"]
        before = signal_prefix(cell, 50)
        cell["metadata"]["group"] = "batch3"
        np.testing.assert_array_equal(before, signal_prefix(cell, 50))

    def test_cross_family_prediction_and_fitting_rejected(self):
        bundle = self.fit()
        self.cells["held_out"]["physical_family"] = list(self.registry["families"])[2]
        with self.assertRaisesRegex(ValueError, "incompatible"):
            predict_baseline(bundle, self.cells["held_out"], 50, [60])
        self.cells["b"]["physical_family"] = list(self.registry["families"])[1]
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.fit()

    def test_reference_must_exist_at_the_fixed_origin(self):
        self.cells["held_out"]["reference_established_source_cycle"] = 65
        with self.assertRaisesRegex(ValueError, "not established"):
            signal_prefix(self.cells["held_out"], 50)

    def test_metrics_and_constant_r2(self):
        metric = regression_metrics([1, 1], [.9, .9])
        self.assertAlmostEqual(metric["mape_percent"], 10)
        self.assertAlmostEqual(metric["rmse_soh"], .1)
        self.assertIsNone(metric["r2"])


if __name__ == "__main__":
    unittest.main()
