import unittest

import numpy as np

from battery_smetimes.models.trajectory import (
    anchored_soh, cellwise_log_rul_radius, log_rul_interval,
)


class JointTrajectoryTest(unittest.TestCase):
    def test_curve_is_positive_and_crosses_at_its_rul(self):
        for shape in (.5, 1, 2, 3):
            horizon = np.array([0, 1, 20, 99.99, 100, 100.01, 150, 1000])
            curve = anchored_soh(.96, 100, horizon, shape)
            self.assertAlmostEqual(curve[0], .96)
            self.assertAlmostEqual(curve[4], .8)
            self.assertTrue(np.all(curve[:4] > .8))
            self.assertTrue(np.all(curve[5:] < .8))
            self.assertTrue(np.all(curve > 0))
            self.assertTrue(np.all(np.diff(curve) < 0))

    def test_invalid_rul_or_future_direction_rejected(self):
        with self.assertRaises(ValueError):
            anchored_soh(.9, 0, [1, 2])
        with self.assertRaises(ValueError):
            anchored_soh(.9, 10, [-1, 2])
        self.assertAlmostEqual(anchored_soh(.79, 10, [0])[0], .8001)

    def test_calibration_counts_cells_not_repeated_observations(self):
        rows = [{"cell_id": str(i), "actual_rul": 10., "predicted_rul": float(i + 1)}
                for i in range(10)]
        radius, info = cellwise_log_rul_radius(rows)
        self.assertEqual(info["rank"], 10)
        duplicate_radius, duplicate_info = cellwise_log_rul_radius(rows * 4)
        self.assertEqual(radius, duplicate_radius)
        self.assertEqual(duplicate_info["calibration_cells"], 10)
        low, high = log_rul_interval(10., radius)
        self.assertLessEqual(low, 10)
        self.assertGreaterEqual(high, 10)
        insufficient, _ = cellwise_log_rul_radius(rows[:8])
        self.assertTrue(np.isinf(insufficient))


if __name__ == "__main__":
    unittest.main()
