"""Unit checks for the fixed XJTU strategy-balancing diagnostic."""

import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_xjtu_group_balance_v2 import group_balanced_arrays


class FakeCell(SimpleNamespace):
    def matrix(self, variant):
        if variant != 'signal_history':
            raise ValueError(variant)
        return self.features


def cell(group, value, review=False):
    return FakeCell(group=group, soh_available=np.array([True, True]),
                    capacity_review_flag=np.array([review, False]),
                    soh=np.array([1., .9]),
                    features=np.full((2, 3), value))


class StrategyWeightsTest(unittest.TestCase):
    def test_each_strategy_and_cell_has_equal_total_mass(self):
        cells = {'a': cell('g1', 1., review=True),
                 'b': cell('g1', 2.), 'c': cell('g2', 3.),
                 'held_out': cell('g3', 999.)}
        x, y, w = group_balanced_arrays(cells, ['a', 'b', 'c'])
        np.testing.assert_array_equal(x[:, 0], [1., 1., 2., 2., 3., 3.])
        np.testing.assert_array_equal(y, [1., .9] * 3)
        self.assertAlmostEqual(w[:2].sum(), .25)
        self.assertAlmostEqual(w[2:4].sum(), .25)
        self.assertAlmostEqual(w[4:].sum(), .5)
        self.assertLess(w[0], w[1])
        self.assertAlmostEqual(w.sum(), 1.)

    def test_single_strategy_rejected(self):
        with self.assertRaises(ValueError):
            group_balanced_arrays({'a': cell('g1', 1.)}, ['a'])


if __name__ == '__main__':
    unittest.main()
