"""Censor-aware MIT ridge regression safeguards."""

import unittest

import numpy as np
from sklearn.linear_model import Ridge

from evaluate_mit_750s_censored_v2 import fit_censored_ridge


class CensoredRidgeTests(unittest.TestCase):
    def test_inactive_censor_bound_does_not_act_like_an_event(self):
        event_x = np.arange(5, dtype=float)[:, None]
        event_life = np.exp(4+.08*np.arange(5))
        censor_x = np.array([[2.5]])
        scaler, coefficients, metadata = fit_censored_ridge(
            event_x, event_life, censor_x, [10.], alpha=1.)
        baseline = Ridge(alpha=1.).fit(scaler.transform(event_x),
                                       np.log(event_life))
        np.testing.assert_allclose(coefficients,
                                   np.r_[baseline.intercept_, baseline.coef_],
                                   atol=1e-5)
        self.assertEqual(metadata['training_censored_cells'], 1)

    def test_survival_lower_bound_only_penalizes_short_prediction(self):
        event_x = np.arange(5, dtype=float)[:, None]
        event_life = np.exp(4+.08*np.arange(5))
        censor_x = np.array([[2.5]])
        scaler, coefficients, _ = fit_censored_ridge(
            event_x, event_life, censor_x, [300.], alpha=1.)
        baseline = Ridge(alpha=1.).fit(scaler.transform(event_x),
                                       np.log(event_life))
        location = scaler.transform(censor_x)
        censored_prediction = coefficients[0]+location[0]@coefficients[1:]
        baseline_prediction = baseline.predict(location)[0]
        self.assertGreater(censored_prediction, baseline_prediction)


if __name__ == '__main__':
    unittest.main()
