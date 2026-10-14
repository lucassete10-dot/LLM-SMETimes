import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_cycle_slope_rul_v2 import corrected_features


class CycleSlopeTests(unittest.TestCase):
    def test_same_soh_change_uses_actual_cycle_span(self):
        cycle = np.r_[np.arange(1, 21), 100]
        cell = SimpleNamespace(cycle=cycle, matrix=lambda variant: np.zeros((21, 3)))
        soh = np.r_[np.ones(20), .9]
        observation = corrected_features(cell, soh, 'last20_observations_per_cycle')
        fixed_cycle = corrected_features(cell, soh, 'last20_cycles')
        self.assertAlmostEqual(observation[-1, -1], -.1 / 99, places=6)
        self.assertAlmostEqual(fixed_cycle[-1, -1], 0, places=6)
        self.assertEqual(observation.shape, fixed_cycle.shape)

    def test_current_prediction_cannot_change_earlier_slope(self):
        cycle = np.arange(1, 31)
        cell = SimpleNamespace(cycle=cycle, matrix=lambda variant: np.zeros((30, 3)))
        soh = np.linspace(1., .9, 30)
        changed = soh.copy()
        changed[-1] -= .1
        for mode in ('last20_observations_per_cycle', 'last20_cycles'):
            np.testing.assert_allclose(corrected_features(cell, soh, mode)[:-1],
                                       corrected_features(cell, changed, mode)[:-1])


if __name__ == '__main__':
    unittest.main()
