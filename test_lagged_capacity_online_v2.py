"""The online-capacity diagnostic must use only a preceding completed cycle."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_lagged_capacity_online_v2 import score_cell


class LaggedCapacityOnlineTest(unittest.TestCase):
    def test_uses_previous_measured_capacity_not_current_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.npz'
            np.savez(path, cycle=np.array([1, 3, 5]),
                     capacity_ah=np.array([2., 1.9, 1.8]),
                     soh=np.array([np.nan, .95, .9]),
                     soh_available=np.array([False, True, True]))
            record = {'cell_id': 'synthetic', 'group': 'test',
                      'labels_file': str(path),
                      'reference_measurement_count': 1,
                      'reference_capacity_ah': 2.}
            result = score_cell(record)
        expected = 50 * (abs(1. - .95) / .95 + abs(.95 - .9) / .9)
        self.assertAlmostEqual(result['mape_percent'], expected)
        self.assertEqual(result['gapped_cycle_rows'], 2)
        self.assertEqual(result['first_previous_diagnostic_cycle'], 1)
        self.assertEqual(result['first_prediction_cycle'], 3)


if __name__ == '__main__':
    unittest.main()
