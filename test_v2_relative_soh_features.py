import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_relative_soh_features_v2 import view_matrix


class RelativeSOHFeatureTests(unittest.TestCase):
    def test_views_are_independent_and_ordered(self):
        cell = SimpleNamespace(raw=np.full((3, 2), 1),
                               delta=np.full((3, 2), 2),
                               history=np.full((3, 2), 3))
        np.testing.assert_array_equal(view_matrix(cell, 'raw_only'), cell.raw)
        np.testing.assert_array_equal(view_matrix(cell, 'delta_only'), cell.delta)
        np.testing.assert_array_equal(view_matrix(cell, 'delta_history')[0], [2, 2, 3, 3])
        with self.assertRaises(ValueError):
            view_matrix(cell, 'unknown')


if __name__ == '__main__':
    unittest.main()
