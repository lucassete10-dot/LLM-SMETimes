import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_mit_inverse_life_v2 import checkpoint_feature, fit_model, predict_endpoint


class MitInverseLifeTests(unittest.TestCase):
    def make_cell(self):
        cycle = np.arange(1, 121)
        raw = np.tile(np.linspace(4.2, 3.8, 51), (len(cycle), 1))
        raw[:, :11] -= cycle[:, None]*.001
        return SimpleNamespace(cycle=cycle, raw=raw, delta=raw-raw[0],
                               reference_count=1)

    def test_checkpoint_feature_is_invariant_to_future_signals(self):
        cell = self.make_cell()
        before = checkpoint_feature(cell, 50)
        changed = self.make_cell()
        changed.raw[50:] = 1000
        changed.delta[50:] = -1000
        np.testing.assert_allclose(before, checkpoint_feature(changed, 50))

    def test_checkpoint_requires_a_recent_observation(self):
        cell = self.make_cell()
        cell.cycle[46:] += 10
        self.assertIsNone(checkpoint_feature(cell, 50))

    def test_inverse_and_log_targets_return_finite_future_endpoints(self):
        cell = self.make_cell()
        x = checkpoint_feature(cell, 50)
        examples = [x, x+.1, x+.2]
        for target in ('inverse', 'log'):
            model = fit_model(examples, [500, 700, 900], target)
            predicted = predict_endpoint(model, x, target, observed_cycle=50)
            self.assertTrue(np.isfinite(predicted))
            self.assertGreater(predicted, 50)
            self.assertLessEqual(predicted, 2500)


if __name__ == '__main__':
    unittest.main()
