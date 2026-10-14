import unittest

import numpy as np

from scripts.diagnose_llm_stage_sampling import stage_drops


class StageSamplingTests(unittest.TestCase):
    def test_constant_curve_has_zero_drop(self):
        x = np.arange(101, 402)
        for robust in (False, True):
            np.testing.assert_array_equal(stage_drops(x, np.ones(len(x)), robust), np.zeros(3))

    def test_offset_does_not_change_degradation(self):
        x = np.arange(101, 402)
        y = 1 - .0001 * x - .000001 * np.maximum(x - 250, 0) ** 2
        for robust in (False, True):
            np.testing.assert_allclose(stage_drops(x, y, robust), stage_drops(x, y + .1, robust), atol=1e-14)

    def test_median_boundary_resists_isolated_spike(self):
        x = np.arange(101, 402)
        y = 1 - .0001 * x
        damaged = y.copy()
        damaged[0] -= .1
        robust_error = np.max(abs(stage_drops(x, damaged, True) - stage_drops(x, y, True)))
        mean_error = np.max(abs(stage_drops(x, damaged) - stage_drops(x, y)))
        self.assertLess(robust_error, .0002)
        self.assertGreater(mean_error, .01)


if __name__ == '__main__':
    unittest.main()
