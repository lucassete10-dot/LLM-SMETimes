import unittest

import numpy as np

from battery_smetimes.features.universal import (causal_shared_features,
                                                   physical_prefix)
from scripts.evaluate_cross_dataset_soh_v3 import subsample_indices


class UniversalFeaturesTest(unittest.TestCase):
    def test_source_layout_and_xjtu_discharge_not_charge(self):
        for domain, width, start, count in (
            ("NASA", 44, 0, 11), ("CALCE", 29, 0, 11),
            ("MIT", 51, 0, 11), ("XJTU", 46, 23, 5),
        ):
            with self.subTest(domain=domain):
                x = np.zeros((2, width), dtype=np.float32)
                x[:, start:start + count] = np.linspace(4.2, 3.2, count)
                x[:, start + count:start + 2 * count] = -2
                if domain != "CALCE":
                    x[:, start + 2 * count:start + 3 * count] = 25
                if domain == "XJTU":
                    x[:, :23] = 999  # bogus charge values must not leak in
                physical, duration = physical_prefix(x, domain, 2.0)
                self.assertAlmostEqual(physical[0, 0], 4.2, places=5)
                self.assertAlmostEqual(physical[0, 1], 3.2, places=5)
                self.assertAlmostEqual(physical[0, 5], 1.0, places=5)
                self.assertAlmostEqual(physical[0, 9], duration / 3600,
                                       places=5)
                self.assertEqual(np.isnan(physical[0, 7]), domain == "CALCE")

    def test_later_diagnostic_cannot_change_earlier_feature(self):
        raw = np.zeros((3, 44), dtype=np.float32)
        raw[:, :11] = np.linspace(4.2, 3.2, 11)
        raw[:, 11:22] = -2
        raw[:, 22:33] = 25
        before = causal_shared_features(raw, "NASA", 2.0, 1)
        raw[2, :11] += .5
        after = causal_shared_features(raw, "NASA", 2.0, 1)
        np.testing.assert_allclose(before[:2], after[:2], equal_nan=True)
        self.assertEqual(before.shape[1], 32)

    def test_train_row_sampling_is_label_independent(self):
        mask = np.array([False, True, True, True, True, True])
        self.assertEqual(subsample_indices(mask, 3).tolist(), [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
