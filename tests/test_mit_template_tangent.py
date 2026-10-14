import unittest
import numpy as np
from battery_smetimes.mit_template_tangent import curve


class TangentTests(unittest.TestCase):
    def test_join_and_slope(self):
        params=np.array([1.,.005,.002,2.]);end=500.;eps=.001
        a,b,c=curve(params,end,[end-eps,end,end+eps])
        np.testing.assert_allclose((b-a)/eps,(c-b)/eps,atol=1e-8,rtol=0)
        np.testing.assert_allclose((c-b)/eps,-(.005+2*.002*2)/100,atol=1e-10,rtol=0)

    def test_inside_original_curve(self):
        params=np.array([1.,.005,.002,2.]);coords=np.array([101.,200.,400.]);h=(coords-100)/100
        np.testing.assert_array_equal(curve(params,500.,coords),1-.005*h-.002*np.maximum(h-2,0)**2)

    def test_not_constant_or_quadratic_after_end(self):
        p=curve([1.,.005,.002,2.],500.,[500.,600.,700.])
        self.assertTrue(np.all(np.diff(p)<0));np.testing.assert_allclose(np.diff(p)[0],np.diff(p)[1],rtol=0,atol=1e-12)

    def test_invalid(self):
        for coords in ([],[100.],[np.inf],[[101.]]):
            with self.assertRaises(ValueError):curve([1.,.005,.002,2.],500.,coords)
        with self.assertRaises(ValueError):curve([1.,.005,.002,2.],np.inf,[101.])


if __name__=='__main__':unittest.main()
