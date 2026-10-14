import unittest

import numpy as np

from audit_xjtu_blended_joint_curve_v2 import coherent_curve
from evaluate_xjtu_age_blend_v2 import blend_endpoints


class AgeBlendTests(unittest.TestCase):
    def test_age_weight_is_known_without_endpoint_label(self):
        result = blend_endpoints(np.array([100.]), np.array([300.]),
                                 np.array([600.]), age_scale=100.)
        np.testing.assert_allclose(result, [500.])

    def test_blended_endpoint_never_precedes_observation(self):
        result = blend_endpoints(np.array([100.]), np.array([0.]),
                                 np.array([0.]), age_scale=100.)
        self.assertEqual(result[0], 100.)

    def test_coherent_curve_crosses_threshold_at_declared_endpoint(self):
        curve = coherent_curve(.95, .8, 100., np.array([0., 25., 50., 100.]))
        self.assertAlmostEqual(curve[0], .95)
        self.assertAlmostEqual(curve[-1], .8)
        self.assertTrue(np.all(np.diff(curve) <= 0))


if __name__ == '__main__':
    unittest.main()
