import unittest

import numpy as np

from xjtu_applicability_v2 import REFERENCE, assess


class XJTUApplicabilityTests(unittest.TestCase):
    def test_reference_and_causal_first_twenty_score(self):
        with np.load(REFERENCE) as saved:
            self.assertEqual(len(saved['cell_ids']), 55)
            self.assertGreater(float(saved['nearest_training_p95']), 0)
        with np.load(r'D:\LLM\XJTU\processed\b1_2C_battery-1.npz') as data:
            raw = data['features'].astype('float32')
        first = assess(raw)
        altered = raw.copy()
        altered[20:] += 100
        second = assess(altered)
        self.assertEqual(first['distance_ratio'], second['distance_ratio'])
        self.assertIn('TRAINING_NEIGHBORHOOD', first['status'])

    def test_short_history_is_not_assessed(self):
        result = assess(np.ones((19, 46), dtype='float32'))
        self.assertEqual(result['status'], 'INSUFFICIENT_HISTORY')
        self.assertIsNone(result['distance_ratio'])


if __name__ == '__main__':
    unittest.main()
