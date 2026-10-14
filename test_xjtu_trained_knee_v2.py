import unittest

import numpy as np

from evaluate_xjtu_trained_knee_v2 import knee_curve, power_curve


class TrainedKneeTests(unittest.TestCase):
    def test_power_curve_crosses_at_rul_and_keeps_linear_tail(self):
        horizon = np.array([0., 25., 50., 100., 150.])
        curve = power_curve(.95, .8, 100., horizon, power=3.)
        self.assertAlmostEqual(curve[0], .95)
        self.assertAlmostEqual(curve[3], .8)
        self.assertAlmostEqual(curve[4], .725)
        self.assertTrue(np.all(np.diff(curve) <= 0))

    def test_knee_curve_reduces_to_linear_when_curvature_zero(self):
        h = np.array([0., 25., 50., 100., 150.])
        np.testing.assert_allclose(knee_curve(.95, .8, 100., h, 0.),
                                   power_curve(.95, .8, 100., h, 1.))

    def test_invalid_power_is_rejected(self):
        with self.assertRaises(ValueError):
            power_curve(.95, .8, 100., np.array([25.]), 5.)


if __name__ == '__main__':
    unittest.main()
