import unittest
from pathlib import Path

import numpy as np

from adapt_xjtu_300_mat_v2 import extract_mat
from predict_xjtu_300_challenger_v2 import predict_series


SAMPLE = Path(r'D:\LLM\XJTU\raw\extracted\Battery Dataset\Batch-5\RW_battery-1.mat')
CACHE = Path('battery_models_v2/xjtu_long_prefix/features.npz')


class XJTU300ChallengerTests(unittest.TestCase):
    def test_signal_only_adapter_matches_development_extraction(self):
        cycle, raw = extract_mat(SAMPLE, 5)
        with np.load(CACHE) as cached:
            np.testing.assert_array_equal(cycle, cached['b5_RW_battery-1_cycle'])
            np.testing.assert_array_equal(raw, cached['b5_RW_battery-1_300s'])
        result = predict_series(raw, cycle)
        np.testing.assert_array_equal(result['cycle'], cycle)
        self.assertTrue(np.isnan(result['soh'][:5]).all())
        self.assertTrue(np.isfinite(result['soh'][5:]).all())
        self.assertIn('NOT EVALUATED', result['rul_status'])

    def test_rejects_wrong_feature_width(self):
        with self.assertRaises(ValueError):
            predict_series(np.zeros((6, 46), dtype='float32'), np.arange(1, 7))


if __name__ == '__main__':
    unittest.main()
