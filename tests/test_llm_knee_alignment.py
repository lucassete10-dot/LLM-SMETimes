import unittest
import numpy as np
from scripts.diagnose_llm_knee_alignment import compare_interval


class KneeAlignmentTests(unittest.TestCase):
    def test_source_cycle_units_and_width(self):
        result = compare_interval(np.zeros(7), {'valid': True, 'lower': 3., 'upper': 4.}, 100)
        self.assertAlmostEqual(result['centre_source_cycle'], 100 + 500 * np.log(2))
        self.assertAlmostEqual(result['transition_width_cycles'], 20 + 100 * np.log(2))
        self.assertEqual(result['outside_interval_cycles'], 0.)

    def test_outside_distance_and_missing_not_zero_success(self):
        p = np.zeros(7)
        result = compare_interval(p, {'valid': True, 'lower': 1., 'upper': 2.}, 100)
        self.assertAlmostEqual(result['outside_interval_cycles'], 500 * np.log(2) - 200)
        result = compare_interval(p, {'valid': False, 'lower': 0., 'upper': 0.}, 100)
        self.assertIsNone(result['outside_interval_cycles'])


if __name__ == '__main__':
    unittest.main()
