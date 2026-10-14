import unittest
from types import SimpleNamespace

import numpy as np

from train_mit_exact80_rul_stage_a import checkpoint, feature_matrix, summarize


class MitExact80RulStageATest(unittest.TestCase):
    def test_checkpoint_uses_only_observed_pre_eol_rows(self):
        cell = SimpleNamespace(
            cycle=np.array([1, 10, 20, 30, 40, 50]),
            soh_available=np.array([False, True, True, True, True, True]),
            capacity_review_flag=np.array([False, False, False, True, False, False]),
        )
        self.assertEqual(checkpoint(cell, 50, .2), 1)
        self.assertEqual(checkpoint(cell, 50, .8), 4)
        self.assertIsNone(checkpoint(cell, 4, .2))

    def test_feature_matrix_has_no_soh_or_capacity_column(self):
        cell = SimpleNamespace(cycle=np.array([1, 2]),
                               soh=np.array([.99, .7]),
                               matrix=lambda variant: np.array([[3, 4], [5, 6]]))
        features = feature_matrix(cell)
        np.testing.assert_array_equal(features, [[1, 3, 4], [2, 5, 6]])

    def test_metrics_report_four_checkpoints_and_rul_error(self):
        rows = [{"fraction": value, "origin_cycle": int(value * 100),
                 "actual_eol_cycle": 100, "cell_id": "x"}
                for value in (.2, .4, .6, .8)]
        result = summarize(rows, [90, 90, 90, 90])
        self.assertEqual(result["20"]["rul_mae_cycles"], 10)
        self.assertEqual(result["80"]["rul_mape_percent"], 50)


if __name__ == "__main__":
    unittest.main()
