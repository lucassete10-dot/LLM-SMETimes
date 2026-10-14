import unittest

import numpy as np

from extract_xjtu_usage_history_v2 import cycle_control, history_features


class XJTUUsageHistoryTests(unittest.TestCase):
    def test_control_reads_early_current_and_temperature_without_capacity(self):
        entry = {'relative_time_min': np.array([0., .5, 1., 0., .5, 0., .5, 1.]),
                 'current_A': np.array([2., 2., 1.9, 0., 0., -3., -3., -2.9]),
                 'temperature_C': np.array([25., 26., 27., 28., 28., 29., 30., 31.])}
        np.testing.assert_allclose(cycle_control(entry), [2., 3., 26.])

    def test_current_and_future_cycle_cannot_change_current_diagnostic_features(self):
        controls = np.array([[1., 2., 25.], [3., 4., 26.],
                             [5., 6., 27.], [7., 8., 28.]], dtype='float32')
        first = history_features(controls, [1, 3])
        changed = controls.copy()
        changed[2:] += 100
        second = history_features(changed, [1, 3])
        self.assertTrue(np.isnan(first[0]).all())
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[1, 0], 2.)  # mean prior charge setpoints, cycles 1 and 2
        with self.assertRaises(ValueError):
            history_features(controls, [3, 3])


if __name__ == '__main__':
    unittest.main()
