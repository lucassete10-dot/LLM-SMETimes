import unittest
import numpy as np
from battery_smetimes.xjtu_personal_shape import basis,target


class PersonalShapeTest(unittest.TestCase):
    def test_future_validation(self):
        for c in ([],[100],[np.inf],[[101]]):
            with self.assertRaises(ValueError):basis(c)

    def test_shape_bounds_and_finite(self):
        c=np.linspace(101,500,90);y=basis(c)@np.array([1.,.01,.025]);a=target(c,y)
        self.assertTrue(np.isfinite(a).all());self.assertTrue(.2<=a[0]<=1.3)
        self.assertTrue(np.all(a[1:]>=0));self.assertTrue(np.all(a[1:]<=5))
        self.assertLess(np.max(abs(basis(c)@a-y)),.005)


if __name__=='__main__':unittest.main()
