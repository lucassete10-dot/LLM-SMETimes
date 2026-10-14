import unittest
import numpy as np
from scripts.audit_xjtu_feature_stability import slope_stats


class FeatureStabilityTest(unittest.TestCase):
    def test_linear_irregular(self):
        c=np.array([1,5,20,50,99.]);v=2+.003*c;r=slope_stats(c,v)
        self.assertAlmostEqual(r['ols'],.3);self.assertAlmostEqual(r['theil_sen'],.3);self.assertFalse(r['flag'])

    def test_missing(self):
        r=slope_stats(np.array([1,5,20.]),np.array([np.nan,1,np.nan]));self.assertIsNone(r['ols']);self.assertEqual(r['finite_points'],1)

    def test_constant(self):
        r=slope_stats(np.arange(5.),np.ones(5));self.assertEqual(r['sensitivity'],0);self.assertFalse(r['flag'])

    def test_float32_matches_float64(self):
        c=np.array([1,5,20,50,99.]);v=np.array([2,2.5,2,3,3.5],dtype=np.float32)
        self.assertEqual(slope_stats(c,v),slope_stats(c,v.astype(float)))


if __name__=='__main__':unittest.main()
