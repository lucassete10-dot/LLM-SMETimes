import unittest

import numpy as np

from evaluate_xjtu_stage_ablation_v2 import assert_reproduced, stage_views


class XJTUStageAblationTests(unittest.TestCase):
    def test_charge_and_discharge_stage_columns_rejoin_exactly(self):
        raw = np.arange(2 * 82, dtype='float32').reshape(2, 82)
        views = stage_views(raw)
        self.assertEqual(views['charge_300s'].shape, (2, 41))
        self.assertEqual(views['discharge_300s'].shape, (2, 41))
        np.testing.assert_array_equal(
            np.concatenate((views['charge_300s'], views['discharge_300s']), axis=1), raw)

    def test_wrong_feature_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '82 columns'):
            stage_views(np.zeros((3, 46), dtype='float32'))

    def test_reproducibility_check_allows_roundoff_but_rejects_metric_drift(self):
        assert_reproduced({'score': [0.1]}, {'score': [0.1 + 1e-13]})
        with self.assertRaisesRegex(ValueError, 'metric differs'):
            assert_reproduced({'score': [0.1]}, {'score': [0.101]})
        with self.assertRaisesRegex(ValueError, 'keys differ'):
            assert_reproduced({'score': [0.1]}, {'scores': [0.1]})


if __name__ == '__main__':
    unittest.main()
