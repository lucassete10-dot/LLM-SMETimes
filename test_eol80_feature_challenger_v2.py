import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_eol80_feature_challenger_v2 import causal_features, checkpoint_score


class FakeCell(SimpleNamespace):
    def matrix(self, variant):
        if variant != 'signal_history':
            raise ValueError(variant)
        return np.column_stack((self.raw, self.raw, self.raw))


class Eol80FeatureTests(unittest.TestCase):
    def make_cell(self):
        cycle = np.arange(1, 31)
        raw = np.column_stack((cycle.astype('float32'),
                               np.sin(cycle).astype('float32')))
        return FakeCell(cycle=cycle, raw=raw, reference_count=1,
                        endpoint_cycle=40, rul_available=np.ones(30, dtype=bool))

    def test_future_signal_changes_do_not_change_past_features(self):
        cell = self.make_cell()
        record = {'charge_rate_c': 2., 'discharge_rate_c': None}
        original = causal_features(cell, record)
        modified = self.make_cell()
        modified.raw[21:] += 1000
        changed = causal_features(modified, record)
        np.testing.assert_allclose(original[:21], changed[:21], equal_nan=True)
        self.assertEqual(original.shape, (30, 22))

    def test_checkpoint_uses_last_available_diagnostic_only(self):
        cell = self.make_cell()
        score = checkpoint_score({'test': cell}, ['test'],
                                 {'test': np.full(30, 18.)}, 20, 35.)
        self.assertEqual(score['cells'], 1)
        self.assertEqual(score['by_cell'][0]['observation_cycle'], 20)
        self.assertEqual(score['mae_cycles'], 2.)
        self.assertEqual(checkpoint_score({'test': cell}, ['test'],
                                         {'test': np.full(30, 18.)}, 50, 35.)['cells'], 0)


if __name__ == '__main__':
    unittest.main()
