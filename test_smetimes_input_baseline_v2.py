import unittest

import numpy as np

from evaluate_smetimes_input_baseline_v2 import input_features


class SyntheticCell:
    def __init__(self, raw):
        self.raw = raw
        self.reference_count = 5


class SMETimesInputBaselineTests(unittest.TestCase):
    def test_flat_features_are_causal(self):
        raw = np.ones((20, 46), dtype='float32')
        first = input_features(SyntheticCell(raw))
        changed = raw.copy()
        changed[12:] += 100.
        second = input_features(SyntheticCell(changed))
        self.assertEqual(first.shape, (20, 282))
        np.testing.assert_array_equal(first[:12], second[:12])


if __name__ == '__main__':
    unittest.main()
