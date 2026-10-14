import unittest

import numpy as np

from evaluate_rul_temporal_stability_v2 import smooth_endpoint


class RULTemporalStabilityTests(unittest.TestCase):
    def test_median_uses_only_past_endpoints(self):
        cycle = np.array([10, 20, 30, 40, 50, 60])
        rul = np.array([90, 80, 70, 40, 50, 40])
        result = smooth_endpoint(cycle, rul, 'last5_records')
        self.assertEqual(result[3], 100.)
        changed = rul.copy()
        changed[-1] = 1000
        np.testing.assert_allclose(result[:-1], smooth_endpoint(cycle, changed,
                                                                 'last5_records')[:-1])

    def test_cycle_window_and_bad_inputs(self):
        cycle = np.array([1, 2, 100])
        rul = np.array([99, 98, 0])
        np.testing.assert_allclose(smooth_endpoint(cycle, rul, 'last50_cycles'), [100, 100, 100])
        with self.assertRaises(ValueError):
            smooth_endpoint([1, 1], [1, 2], 'raw')


if __name__ == '__main__':
    unittest.main()
