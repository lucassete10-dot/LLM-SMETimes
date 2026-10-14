import unittest
import numpy as np
from battery_smetimes.neighbor_curve import extrapolate,predict

class NeighborCurveTest(unittest.TestCase):
    def test_linear_and_extrapolation(self):
        np.testing.assert_allclose(extrapolate([101,201,301],[1,.9,.8],[101,151,401]),[1,.95,.7],rtol=0,atol=1e-12)
    def test_no_endpoint_plateau(self):
        p=extrapolate([101,201,301],[1,.98,.8],[301,401]);self.assertLess(p[1],p[0])
    def test_family_rejection(self):
        with self.assertRaises(ValueError):predict({'family':'A'},{'physical_family':'B'},[101],1)

if __name__=='__main__':unittest.main()
