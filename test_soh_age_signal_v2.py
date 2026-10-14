import unittest
from types import SimpleNamespace

import numpy as np

from audit_mit_causal_soh_smoothing_v2 import past_median
from audit_soh_error_structure_v2 import local_interpolation_error
from evaluate_mit_partial_qv_soh_v2 import GRID, qv_from_300s_sample
from evaluate_mit_decaying_anchor_v2 import decaying_anchor
from evaluate_mit_policy_soh_v2 import policy_matrix
from evaluate_soh_age_signal_v2 import age_matrix


class CausalSohFeatureTests(unittest.TestCase):
    def make_cell(self):
        cycle = np.arange(1, 9)
        raw = np.tile(np.linspace(3.55, 3.1, 51), (len(cycle), 1))
        raw[:, :11] -= .001*cycle[:, None]
        return SimpleNamespace(cycle=cycle, raw=raw, delta=raw-raw[0])

    def test_age_matrix_prefix_ignores_future_rows(self):
        cell = self.make_cell()
        first = age_matrix(cell)[:4]
        changed = self.make_cell()
        changed.raw[4:] = 999
        changed.delta[4:] = 999
        np.testing.assert_allclose(first, age_matrix(changed)[:4], equal_nan=True)

    def test_partial_qv_uses_only_voltage_and_current_samples(self):
        row = np.zeros(51)
        row[:11] = np.linspace(3.55, 3.1, 11)
        row[11:22] = -4.4
        a = qv_from_300s_sample(row)
        changed = row.copy()
        changed[22:] = 999
        np.testing.assert_allclose(a, qv_from_300s_sample(changed))
        self.assertEqual(a.shape, GRID.shape)
        self.assertTrue(np.isfinite(a).all())

    def test_smoother_ignores_future_predictions(self):
        cycle = np.arange(1, 9)
        values = np.arange(8, dtype=float)
        first = past_median(cycle, values, 5)[:4]
        changed = values.copy()
        changed[4:] = 999
        np.testing.assert_allclose(first, past_median(cycle, changed, 5)[:4])

    def test_decaying_anchor_ignores_future_predictions(self):
        cycle = np.arange(1, 9)
        values = 0.99 - .01*cycle
        first = decaying_anchor(cycle, values, 2, 50)[:4]
        changed = values.copy()
        changed[4:] = 99
        np.testing.assert_allclose(first, decaying_anchor(cycle, changed, 2, 50)[:4])

    def test_policy_features_are_known_cell_metadata(self):
        cell = self.make_cell()
        record = {'charge_rate_steps_c': [4.0, 5.0]}
        matrix = policy_matrix(cell, record)
        np.testing.assert_allclose(matrix[:, -3:], np.tile([4., 5., 1.],
                                                            (len(cell.cycle), 1)))
        changed = self.make_cell()
        changed.raw[4:] = 999
        changed.delta[4:] = 999
        np.testing.assert_allclose(matrix[:4], policy_matrix(changed, record)[:4],
                                   equal_nan=True)

    def test_neighbor_interpolation_is_zero_for_linear_target(self):
        result = local_interpolation_error(np.arange(1, 11),
                                           1-.01*np.arange(1, 11))
        self.assertAlmostEqual(result['mape_percent'], 0)


if __name__ == '__main__':
    unittest.main()
