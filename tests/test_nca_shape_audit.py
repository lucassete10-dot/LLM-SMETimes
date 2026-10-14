import unittest
import numpy as np
from scripts.audit_nca_curve_shape import shape


class ShapeTests(unittest.TestCase):
    def test_plateau_exposed_despite_direction(self):
        x=np.arange(101,106);y=np.linspace(.95,.8,5);p=np.array([.94,.9,.86,.86,.86])
        report=shape(x,y,p)
        self.assertEqual(report['predicted_late_change_pp'],0)
        self.assertEqual(report['late_change_ratio'],0)
        self.assertAlmostEqual(report['endpoint_signed_bias_pp'],6)

    def test_wrong_direction_exposed(self):
        x=np.arange(101,106);y=np.linspace(.95,.8,5);p=np.linspace(.9,.98,5)
        self.assertTrue(shape(x,y,p)['opposite_late_direction'])

    def test_tiny_truth_change_not_unstable_ratio(self):
        x=np.arange(101,106);y=np.linspace(.95,.949,5);p=np.ones(5)*.94
        self.assertIsNone(shape(x,y,p)['late_change_ratio'])


if __name__=='__main__':unittest.main()
