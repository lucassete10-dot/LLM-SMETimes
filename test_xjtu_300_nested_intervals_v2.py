"""Causality and accounting checks for XJTU nested SOH interval audit."""

import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_xjtu_300_nested_intervals_v2 import coverage, inner_splits, labeled_errors


class NestedIntervalsTest(unittest.TestCase):
    def test_inner_splits_never_use_outer_validation(self):
        cells = {f'g{group}_{index}': SimpleNamespace(group=f'g{group}')
                 for group in range(4) for index in range(2)}
        cells['outer'] = SimpleNamespace(group='g4')
        ids = [key for key in cells if key != 'outer']
        splits = list(inner_splits(cells, ids))
        self.assertEqual(len(splits), 4)
        self.assertEqual({held for held, _, _ in splits}, {'g0', 'g1', 'g2', 'g3'})
        for held, train, validation in splits:
            self.assertNotIn('outer', train + validation)
            self.assertTrue(all(cells[key].group != held for key in train))
            self.assertTrue(all(cells[key].group == held for key in validation))
            self.assertEqual(set(train) | set(validation), set(ids))

    def test_coverage_and_errors_use_only_scored_observations(self):
        cell = SimpleNamespace(group='g0', soh_available=np.array([False, True, True]),
                               soh=np.array([1., .9, .8]), cycle=np.array([1, 2, 3]),
                               endpoint_cycle=3)
        predictions = {'a': np.array([.2, .95, .7])}
        errors, ids, groups = labeled_errors({'a': cell}, predictions, ['a'])
        np.testing.assert_allclose(errors, [.05, -.1])
        self.assertEqual(ids.tolist(), ['a', 'a'])
        self.assertEqual(groups.tolist(), ['g0', 'g0'])
        result = coverage({'a': cell}, predictions, ['a'], .075)
        self.assertAlmostEqual(result['cell_macro_coverage'], .5)
        self.assertAlmostEqual(result['observation_coverage'], .5)


if __name__ == '__main__':
    unittest.main()
