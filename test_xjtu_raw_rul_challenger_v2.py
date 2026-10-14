import unittest

import numpy as np

from predict_xjtu_raw_rul_challenger_v2 import predict_series


class XJTURawRULChallengerTests(unittest.TestCase):
    def test_label_free_inference_masks_unavailable_early_rul(self):
        cycle = np.arange(1, 24)
        raw = np.ones((len(cycle), 46), dtype='float32')
        result = predict_series(raw, cycle)
        self.assertTrue(np.isnan(result['rul_cycles'][:15]).all())
        self.assertTrue(np.isfinite(result['rul_cycles'][15:]).all())
        self.assertTrue((result['rul_cycles'][15:] >= 0).all())

    def test_feature_width_is_checked(self):
        with self.assertRaisesRegex(ValueError, '46-column'):
            predict_series(np.ones((20, 45), dtype='float32'), np.arange(1, 21))


if __name__ == '__main__':
    unittest.main()
