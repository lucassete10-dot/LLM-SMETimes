import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluate_development_intervals_v2 import evaluate
from train_trajectory_v2 import future_pairs, predict_trajectory, threshold_soh


class TrajectoryAndIntervalTests(unittest.TestCase):
    def test_dataset_specific_thresholds(self):
        for domain, expected in [('NASA', .8), ('CALCE', .8),
                                 ('MIT', .8), ('XJTU', .8)]:
            reference = 1.1 if domain == 'MIT' else 2.0
            cell = SimpleNamespace(dataset=domain, reference_capacity_ah=reference)
            self.assertAlmostEqual(threshold_soh(cell), expected)

    def test_future_pairs_do_not_cross_endpoint(self):
        cycle = np.arange(1, 121)
        cell = SimpleNamespace(cycle=cycle, soh_available=np.ones(120, dtype=bool),
                               reference_count=1, endpoint_cycle=100)
        current, future = future_pairs(cell, 50)
        self.assertTrue(len(current) > 0)
        self.assertTrue(np.all(future > current))
        self.assertTrue(np.all(cycle[future] <= 100))

    def test_projection_uses_current_rate_only(self):
        cell = SimpleNamespace()
        soh = np.array([1., .95, .9])
        rate = np.array([.002, .001, .003])
        np.testing.assert_allclose(predict_trajectory(cell, soh, rate, 1, [0, 10, 20]),
                                   [.95, .94, .93], rtol=1e-6)

    def test_interval_radius_excludes_current_split(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'oof.npz'
            np.savez(path, target=np.zeros(4), estimate=np.array([1., 1., 10., 10.]),
                     held_out=np.array(['a', 'a', 'b', 'b']),
                     cell_id=np.array(['c1', 'c1', 'c2', 'c2']), cycle=np.arange(1, 5))
            report = evaluate(path, .9)
        by_fold = {row['held_out']: row for row in report['folds']}
        self.assertEqual(by_fold['a']['calibration_radius'], 10.)
        self.assertEqual(by_fold['b']['calibration_radius'], 1.)
        self.assertEqual(by_fold['a']['observation_coverage'], 1.)
        self.assertEqual(by_fold['b']['observation_coverage'], 0.)


if __name__ == '__main__':
    unittest.main()
