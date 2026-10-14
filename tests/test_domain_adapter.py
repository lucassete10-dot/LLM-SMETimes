import unittest

import numpy as np

from battery_smetimes.models.domain_adapter import (apply_offset,
                                                      fit_cell_balanced_offset)
from scripts.evaluate_cross_dataset_calibration import choose_calibration


class DomainAdapterTest(unittest.TestCase):
    def test_cell_balance_not_diagnostic_pooling(self):
        calibration = {
            "long": (np.zeros(100), np.full(100, .1)),
            "short": (np.zeros(2), np.full(2, .3)),
        }
        self.assertAlmostEqual(fit_cell_balanced_offset(calibration), .2)
        np.testing.assert_allclose(apply_offset([.7, .8], .2), [.9, 1.0])

    def test_calibration_cells_are_disjoint_and_label_budget_exact(self):
        predicted = {str(i): (np.full(100, .8), np.full(100, .9))
                     for i in range(12)}
        calibration, evaluation, budget, total = choose_calibration(
            predicted, .05, 2026)
        self.assertEqual((budget, total), (60, 1200))
        self.assertEqual(sum(len(pair[0]) for pair in calibration.values()), 60)
        self.assertFalse(set(calibration) & set(evaluation))
        self.assertEqual(len(calibration) + len(evaluation), 12)

    def test_fixed_early_prefix_uses_more_cells_but_no_later_labels(self):
        predicted = {str(i): (np.arange(200, dtype=float), np.ones(200))
                     for i in range(12)}
        calibration, evaluation, budget, total = choose_calibration(
            predicted, .05, 2026, prefix_limit=100)
        self.assertEqual((budget, total), (120, 2400))
        self.assertEqual(sum(len(pair[0]) for pair in calibration.values()), 120)
        self.assertTrue(all(np.max(pair[0]) < 100 for pair in calibration.values()))
        self.assertFalse(set(calibration) & set(evaluation))

    def test_invalid_calibration_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid calibration"):
            fit_cell_balanced_offset({"bad": ([.8], [np.nan])})


if __name__ == "__main__":
    unittest.main()
