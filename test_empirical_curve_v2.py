import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_empirical_curve_v2 import (design, early_features, first_origin,
                                         predict_curve, score_cell)


class Identity:
    def transform(self, x):
        return x


class Model:
    # Three transformed features plus intercept in each fade component.
    coef_ = np.array([.1, 0., 0., 0., .01, 0., 0., 0.])


class EmpiricalCurveTests(unittest.TestCase):
    def test_origin_uses_observed_length_not_future_label_masks(self):
        cell = SimpleNamespace(cycle=np.arange(20), reference_count=3,
                               soh_available=np.zeros(20, dtype=bool),
                               rul_available=np.zeros(20, dtype=bool))
        self.assertEqual(first_origin(cell), 15)

    def test_design_has_zero_drop_at_zero_horizon(self):
        x = design(np.array([2.]), np.array([0., 100.]), 100.)
        self.assertTrue(np.all(x[0] == 0))

    def test_relative_features_use_only_origin_and_prior_history(self):
        cell = SimpleNamespace(cycle=np.arange(20),
                               delta=np.arange(20, dtype=float)[:, None],
                               history=np.arange(20, dtype=float)[:, None])
        before = early_features(cell, 15, 'relative')
        cell.delta[16:] = 1000
        cell.history[16:] = 1000
        np.testing.assert_array_equal(before, early_features(cell, 15, 'relative'))

    def test_forecast_is_bounded_monotone(self):
        cell = SimpleNamespace(cycle=np.arange(20), reference_count=1,
                               matrix=lambda _: np.zeros((20, 1)))
        bundle = {'imputer': Identity(), 'scaler': Identity(),
                  'pca': Identity(), 'model': Model(), 'scale': 100.,
                  'feature_mode': 'full'}
        curve, coefficients = predict_curve(bundle, cell, .95, 15, 250)
        self.assertTrue(np.all(np.diff(curve) <= 0))
        self.assertTrue(np.all((curve >= 0) & (curve <= .95)))
        np.testing.assert_allclose(coefficients, [.1, .01])

    def test_non_crossing_endpoint_is_scored_as_failure(self):
        cell = SimpleNamespace(cell_id='example', group='g', dataset='XJTU',
                               reference_capacity_ah=2., cycle=np.arange(19),
                               endpoint_cycle=18, soh_available=np.ones(19, dtype=bool),
                               soh=np.linspace(1., .8, 19))
        result = score_cell(cell, np.full(10, .95), 15, np.array([0., 0.]))
        self.assertFalse(result['crossed_within_grid'])
        self.assertEqual(result['predicted_endpoint_cycle'], 25)
        self.assertGreater(result['eol_ape_percent'], 0)


if __name__ == '__main__':
    unittest.main()
