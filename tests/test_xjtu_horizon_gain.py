import unittest
import numpy as np
from battery_smetimes.xjtu_horizon_gain import phi,fit_coefficients


class HorizonGainTests(unittest.TestCase):
    def test_phi(self):
        c=np.array([101.,200.,10000.]);f=phi(c)
        np.testing.assert_allclose(f,(c-100)/c,atol=1e-15,rtol=0)
        self.assertTrue(np.all(f>0) and np.all(f<1) and np.all(np.diff(f)>0))

    def test_invalid(self):
        for c in ([],[100.],[np.inf],[[101.]]):
            with self.assertRaises(ValueError):phi(c)

    def test_identity(self):
        c=np.arange(101.,301.);p=1-(c-101)*.0005
        a,info=fit_coefficients([dict(cycle=c,truth=p,prediction=p)],.01)
        np.testing.assert_allclose(a,[1,0],rtol=0,atol=1e-5)
        self.assertTrue(info['weighted_design_rows']>2)


if __name__=='__main__':unittest.main()
