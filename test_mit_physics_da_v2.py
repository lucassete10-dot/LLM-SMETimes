"""Safety checks for the first physics/domain-alignment experiment."""

import unittest

import numpy as np
import torch

from evaluate_mit_physics_da_v2 import (coral_penalty, physics_penalty,
                                        prepare_fold)
from audit_mit_physics_da_v2 import causal_upward_cap
from audit_mit_target_cell_calibration_v2 import evaluate_calibration


class PhysicsDaTest(unittest.TestCase):
    def test_physics_penalty_allows_small_regeneration(self):
        current = torch.tensor([.999, 1.01])
        previous = torch.tensor([1., 1.])
        gap = torch.tensor([1., 2.])
        self.assertAlmostEqual(float(physics_penalty(current, previous, gap)), .003)

    def test_coral_is_zero_for_identical_source_embeddings(self):
        hidden = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        groups = torch.tensor([0, 0, 1, 1])
        self.assertAlmostEqual(float(coral_penalty(hidden, groups)), 0.)

    def test_preprocessing_does_not_fit_held_out_batch(self):
        training = ['source']
        a = {'source': np.array([[0.], [1.], [2.]]),
             'heldout': np.array([[100.]])}
        b = {'source': a['source'], 'heldout': np.array([[100000.]])}
        first = prepare_fold(None, a, training, ['heldout'])
        second = prepare_fold(None, b, training, ['heldout'])
        np.testing.assert_allclose(first['source'], second['source'])
        np.testing.assert_allclose(first['source'].mean(), 0., atol=1e-6)

    def test_causal_cap_does_not_see_future_predictions(self):
        cycles = np.array([1, 2, 4, 5])
        original = np.array([1., .95, 1., .94])
        changed = original.copy()
        changed[3] = 5.
        np.testing.assert_allclose(
            causal_upward_cap(cycles, original)[:3],
            causal_upward_cap(cycles, changed)[:3])
        self.assertLessEqual(causal_upward_cap(cycles, original)[2], .954)

    def test_target_calibration_keeps_evaluation_cell_disjoint(self):
        cells = {'cal': {'target': np.array([.9, .8]),
                         'estimate': np.array([.8, .7])},
                 'eval': {'target': np.array([.9, .8]),
                          'estimate': np.array([.8, .7])}}
        mape, correction = evaluate_calibration(cells, ['cal'], ['eval'], 1.)
        self.assertAlmostEqual(correction, .1)
        self.assertAlmostEqual(mape, 0.)
        with self.assertRaises(ValueError):
            evaluate_calibration(cells, ['cal'], ['cal'], 1.)


if __name__ == '__main__':
    unittest.main()
