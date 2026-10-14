import unittest

import numpy as np

from evaluate_interval_robustness_v2 import radii


class IntervalRobustnessTests(unittest.TestCase):
    def test_group_radius_uses_training_groups_not_one_pooled_cycle_count(self):
        errors = np.array([1.]*10 + [10.])
        cells = np.array(['a']*10 + ['b'])
        groups = np.array(['g1']*10 + ['g2'])
        result = radii(errors, cells, groups)
        self.assertEqual(result['worst_training_group_p90'], 10.)
        self.assertEqual(result['cell_p90_then_p90'], 10.)
        self.assertEqual(result['pooled_cycle_p90'], 1.)
        with self.assertRaisesRegex(ValueError, 'differ in length'):
            radii(errors, cells[:1], groups)


if __name__ == '__main__':
    unittest.main()
