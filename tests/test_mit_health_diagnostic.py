from types import SimpleNamespace
import unittest

import numpy as np

from scripts.diagnose_mit_health_rul import observed_health_features
from scripts.diagnose_mit_estimated_health_rul import predicted_health_features


class HealthDiagnosticTest(unittest.TestCase):
    def test_known_decline_uses_actual_cycles_without_future_labels(self):
        cycle = np.array([1, 6, 11, 16, 21, 26, 31])
        cell = SimpleNamespace(cycle=cycle, soh=1 - cycle * .001,
                               soh_available=np.ones(len(cycle), dtype=bool),
                               capacity_review_flag=np.zeros(len(cycle), dtype=bool))
        original = observed_health_features(cell)
        self.assertAlmostEqual(float(original[3, 3]), -.001, places=6)
        cell.soh[4:] = -999
        cell.capacity_review_flag[4:] = True
        changed = observed_health_features(cell)
        np.testing.assert_array_equal(original[:4], changed[:4])

    def test_predicted_history_does_not_read_measured_capacity_flags_or_soh(self):
        cell = SimpleNamespace(cycle=np.arange(1, 12), reference_count=2)
        estimate = 1 - cell.cycle * .001
        before = predicted_health_features(cell, estimate)
        cell.soh = np.full(11, -999.)
        cell.capacity_review_flag = np.ones(11, dtype=bool)
        cell.soh_available = np.zeros(11, dtype=bool)
        np.testing.assert_array_equal(before, predicted_health_features(cell, estimate))


if __name__ == "__main__":
    unittest.main()
