import unittest

import numpy as np

from evaluate_xjtu_applicability_cv_v2 import (distance_ratio,
                                                fit_distance_reference, summarize)


class XJTUApplicabilityCVTests(unittest.TestCase):
    def test_distance_reference_uses_only_supplied_training_vectors(self):
        training = np.arange(6 * 46, dtype='float32').reshape(6, 46)
        fitted = fit_distance_reference(training)
        self.assertEqual(fitted[2].shape, (6, 46))
        self.assertGreater(fitted[3], 0)
        self.assertGreater(distance_ratio(training[0] + 1000, fitted), 1)

    def test_empty_summary_does_not_fabricate_error(self):
        self.assertEqual(summarize([]), {'cells': 0, 'soh_mae': None,
                                         'rul_mae_cycles': None})


if __name__ == '__main__':
    unittest.main()
