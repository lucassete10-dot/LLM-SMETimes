import unittest

import numpy as np

from evaluate_soh_anchor_v2 import anchored_prediction


class AnchorCorrectionTests(unittest.TestCase):
    def test_first_five_reference_median_and_causality(self):
        prediction = np.array([.92, .94, .96, .98, 1., .85, .80])
        corrected = anchored_prediction(prediction, 5, 1.)
        self.assertAlmostEqual(np.median(corrected[:5]), 1.)
        self.assertAlmostEqual(corrected[5], .89)
        changed_future = prediction.copy()
        changed_future[6] = .2
        np.testing.assert_allclose(corrected[:6],
                                   anchored_prediction(changed_future, 5, 1.)[:6])

    def test_zero_strength_and_invalid_reference(self):
        np.testing.assert_allclose(anchored_prediction([.8, .7], 1, 0), [.8, .7])
        with self.assertRaises(ValueError):
            anchored_prediction([.8, .7], 3, 1)
        with self.assertRaises(ValueError):
            anchored_prediction([np.nan, .7], 1, 1)


if __name__ == '__main__':
    unittest.main()
