import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_multihorizon_trajectory_v2 import append_horizon, training_parts


class MultihorizonTests(unittest.TestCase):
    def test_horizon_column_and_invalid_span(self):
        result = append_horizon(np.ones((2, 3)), [50, 100])
        np.testing.assert_allclose(result[:, -2], [50, 100])
        np.testing.assert_allclose(result[:, -1], np.log1p([50, 100]))
        with self.assertRaises(ValueError):
            append_horizon(np.ones((2, 3)), [50, 0])

    def test_training_pairs_stop_before_endpoint(self):
        # A minimal cell with no eligible future pair must not fabricate one.
        cycle = np.arange(1, 21)
        cell = SimpleNamespace(cycle=cycle, reference_count=1,
                               endpoint_cycle=18, soh=np.ones(20),
                               soh_available=np.ones(20, dtype=bool),
                               capacity_review_flag=np.zeros(20, dtype=bool))
        current, future = __import__('train_trajectory_v2').future_pairs(cell, 50)
        self.assertEqual(len(current), 0)


if __name__ == '__main__':
    unittest.main()
