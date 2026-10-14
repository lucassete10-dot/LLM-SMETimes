import unittest
import numpy as np
from battery_smetimes.xjtu_balanced_acceleration import psi,fit_coefficients


class BalancedAccelerationTest(unittest.TestCase):
    def test_basis(self):
        c=np.array([101.,200.,400.,10000.]);h=(c-100)/100
        np.testing.assert_allclose(psi(c),h-1+1/(1+h),rtol=0,atol=1e-14)
        self.assertTrue(np.all(np.diff(psi(c))>0))

    def test_invalid(self):
        for c in ([],[100.],[np.inf],[[101.]]):
            with self.assertRaises(ValueError):psi(c)

    def test_identity(self):
        c=np.linspace(101,500,80);p=1.-.01*(c-100)/100
        coef,_=fit_coefficients([dict(cycle=c,truth=p,prediction=p)],.01)
        np.testing.assert_allclose(coef,[1.,0.],atol=1e-5,rtol=0)


if __name__=='__main__':unittest.main()
