import unittest
from types import SimpleNamespace

import numpy as np

from train_shared_smetimes_v2 import physical_features
from train_xjtu_relative_voltage_v2 import first_crossing


class FeatureExperimentTests(unittest.TestCase):
    def test_xjtu_charge_discharge_mapping_and_causality(self):
        raw = np.zeros((8, 46), dtype='float32')
        raw[:, :5] = [3.5, 3.55, 3.6, 3.65, 3.7]
        raw[:, 23:28] = [3.8, 3.7, 3.6, 3.5, 3.4]
        cell = SimpleNamespace(dataset='XJTU', raw=raw, reference_count=5)
        first = physical_features(cell)
        self.assertEqual(first.shape, (8, 90))
        np.testing.assert_allclose(first[0, :5], [3.8, 3.7, 3.6, 3.5, 3.4])
        np.testing.assert_allclose(first[0, 15:20], [3.5, 3.55, 3.6, 3.65, 3.7])
        changed = raw.copy()
        changed[7, :5] += 1
        cell_changed = SimpleNamespace(dataset='XJTU', raw=changed, reference_count=5)
        np.testing.assert_allclose(first[:7], physical_features(cell_changed)[:7])

    def test_relative_voltage_crossing_and_missing(self):
        voltage = np.array([3.5, 3.55, 3.6, 3.65, 3.7], dtype='float32')
        self.assertAlmostEqual(first_crossing(voltage, .05), 30, places=3)
        self.assertAlmostEqual(first_crossing(voltage, .10), 60, places=3)
        self.assertTrue(np.isnan(first_crossing(voltage, .25)))


if __name__ == '__main__':
    unittest.main()
