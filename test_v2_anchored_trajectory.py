import unittest

import numpy as np

from evaluate_anchored_trajectory_v2 import anchored_soh


class AnchoredTrajectoryDiagnosticTests(unittest.TestCase):
    def test_path_starts_at_soh_and_hits_threshold_at_selected_rul(self):
        soh = np.array([1.0, .95])
        selected_rul = np.array([100., 200.])
        local_rul = np.array([500., 50.])
        np.testing.assert_allclose(anchored_soh(soh, .8, selected_rul, local_rul,
                                               np.zeros(2)), soh)
        np.testing.assert_allclose(anchored_soh(soh, .8, selected_rul, local_rul,
                                               selected_rul), [.8, .8], atol=1e-6)

    def test_nonpositive_rul_is_already_at_threshold(self):
        estimate = anchored_soh(np.array([.9]), .8, np.array([0.]),
                                np.array([100.]), np.array([25.]))
        np.testing.assert_allclose(estimate, [.8])


if __name__ == '__main__':
    unittest.main()
