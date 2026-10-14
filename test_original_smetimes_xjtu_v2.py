import unittest

import numpy as np

from evaluate_original_smetimes_xjtu_v2 import historical_features, sample_indices


class OriginalSMETimesXJTUTests(unittest.TestCase):
    def test_pretraining_sample_selection_is_bounded_and_unique(self):
        self.assertEqual(len(sample_indices(100, 24)), 24)
        self.assertEqual(len(sample_indices(8, 24)), 8)
        self.assertEqual(sample_indices(100, 24)[0], 0)
        self.assertEqual(sample_indices(100, 24)[-1], 99)

    def test_history_does_not_use_future_embeddings(self):
        z = np.arange(40, dtype='float32').reshape(20, 2)
        first = historical_features(z, 5)
        changed = z.copy()
        changed[12:] += 1000
        second = historical_features(changed, 5)
        np.testing.assert_array_equal(first[:12], second[:12])


if __name__ == '__main__':
    unittest.main()
