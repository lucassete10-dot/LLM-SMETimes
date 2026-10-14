import unittest
import numpy as np
from battery_smetimes.xjtu_personal_residual import basis,target


class PersonalResidualTest(unittest.TestCase):
    def test_anchor(self):np.testing.assert_array_equal(basis([101.]),[[0.,0.]])

    def test_invalid(self):
        for c in ([],[100],[np.inf],[[101]]):
            with self.assertRaises(ValueError):basis(c)

    def test_zero_residual(self):
        c=np.linspace(101,500,60);p=1-.02*(c-101)/100
        np.testing.assert_array_equal(target(dict(cycle=c,truth=p,prediction=p)),[0.,0.])

    def test_recovers_sign(self):
        c=np.linspace(101,500,60);p=1-.02*(c-101)/100;y=p-.01*basis(c)[:,0]
        a=target(dict(cycle=c,truth=y,prediction=p));self.assertLess(a[0],0);self.assertTrue(np.all(abs(a)<=.1))


if __name__=='__main__':unittest.main()
