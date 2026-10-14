import unittest

import numpy as np

from battery_smetimes.forecast import prefix_vector, type_key, query_matrix, charge_rates, fit_curve_model, predict_curve
from scripts.train_same_type_forecast import make_splits
from scripts.audit_mit_discharge_integrity import discontinuity
from scripts.train_nasa_voltage_forecast import voltage_prefix


class SameTypeForecastTests(unittest.TestCase):
    def test_nasa_voltage_candidate_ignores_future_and_rejects_missing_voltage(self):
        cycle = np.arange(1, 151)
        raw = np.full((150, 44), 3.6)
        raw[100:] = np.inf
        self.assertAlmostEqual(voltage_prefix({"cycle": cycle, "raw": raw}, 100), 3.6)
        raw[97:100, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            voltage_prefix({"cycle": cycle, "raw": raw}, 100)

    def test_interruption_rule_uses_time_and_current_not_capacity(self):
        self.assertIsNone(discontinuity([0, .1, .2], [-4, -4, -4])[0])
        self.assertIsNone(discontinuity([0, .1, .1 - 1e-12, .2], [-4] * 4)[0])
        self.assertEqual(discontinuity([0, .1, 500], [-4, -4, -4])[0],
                         "gap_over_60s_between_adjacent_4C_discharge_samples")
        self.assertEqual(discontinuity([0, .1, 500], [-4, 0, -4])[0],
                         "4C_discharge_episode_spans_over_1_hour_interrupted_protocol")
        self.assertIsNone(discontinuity([0, .1, 5], [-4, 0, -4])[0])
        self.assertEqual(discontinuity([0, .1, 0], [-4, -4, -4])[0], "nonmonotone_time")

    def test_future_signals_cannot_change_prefix(self):
        metadata = {"dataset": "NASA", "discharge_cutoff_v": 2.5}
        cycle = np.arange(1, 151)
        x = np.random.default_rng(1).normal(size=(150, 44))
        expected = prefix_vector(cycle, x, metadata, 50, [type_key(metadata)])
        x[50:] = np.inf
        np.testing.assert_array_equal(expected, prefix_vector(cycle, x, metadata, 50, [type_key(metadata)]))

    def test_sparse_time_and_insufficient_prefix(self):
        metadata = {"dataset": "XJTU", "group": "batch6"}
        cycle = np.arange(2, 102, 11)
        x = np.tile(cycle[:, None] / 100, (1, 46))
        with self.assertRaisesRegex(ValueError, "insufficient"):
            prefix_vector(cycle, x, metadata, 20, [type_key(metadata)])
        vector = prefix_vector(cycle, x, metadata, 100, [type_key(metadata)])
        np.testing.assert_allclose(vector[3 * 46:4 * 46], 1, rtol=1e-5)

    def test_policy_without_trailing_unit(self):
        metadata = {"dataset": "MIT", "group": "batch2", "charge_protocol": "4C(31%)-5", "charge_rate_steps_c": [4]}
        self.assertEqual(charge_rates(metadata), [4, 5])
        self.assertEqual(type_key(metadata), "MIT/batch2/increasing")

    def test_invalid_future_query(self):
        with self.assertRaises(ValueError):
            query_matrix(np.zeros(2), 100, [100, 101])

    def test_training_does_not_read_unlisted_cell_labels(self):
        metadata = {"dataset": "NASA", "discharge_cutoff_v": 2.5}
        cycle = np.arange(1, 151)
        cells = {key: {"metadata": metadata, "cycle": cycle[:100],
            "raw": np.tile(cycle[:100, None] / 100, (1, 44)), "label_cycle": cycle,
            "soh": 1 - cycle * rate} for key, rate in (("a", .001), ("b", .0012), ("heldout", .002))}
        first = fit_curve_model(cells, ["a", "b"], 4)
        cells["heldout"]["soh"][:] = -999
        second = fit_curve_model(cells, ["a", "b"], 4)
        c = cells["heldout"]
        args = (c["cycle"], c["raw"], metadata, 100, [110, 130, 150])
        # Parallel tree reduction can differ at machine epsilon.
        np.testing.assert_allclose(predict_curve(first, *args)[0], predict_curve(second, *args)[0], rtol=0, atol=1e-12)
        self.assertEqual(first["train_ids"], ["a", "b"])

    def test_split_is_whole_cell_and_has_seen_type(self):
        cells = {}
        for domain, n in (("NASA", 4), ("MIT", 10), ("XJTU", 8)):
            for index in range(n):
                cells[f"{domain}{index}"] = {"metadata": {"dataset": domain, "group": "batch1",
                    "charge_protocol": "4C(80%)-4C", "charge_rate_steps_c": [4, 4]}}
        for split in make_splits(cells):
            self.assertFalse(set(split["train"]) & set(split["test"]))
            self.assertFalse(set(split["validation"]) & set(split["test"]))


if __name__ == "__main__":
    unittest.main()
