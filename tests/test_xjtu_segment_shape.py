import unittest
import numpy as np
from battery_smetimes.xjtu_segment_shape import evaluate_coefficients,predict

class ShapeTests(unittest.TestCase):
    def test_linear_actual_cycles(self):
        np.testing.assert_allclose(evaluate_coefficients([1,.02],[150,200,400]),[.99,.98,.94])
    def test_quadratic_accelerates(self):
        p=evaluate_coefficients([1,.01,.01],[200,300,400])
        self.assertGreater(p[1]-p[2],p[0]-p[1])
    def test_not_clipped(self):
        self.assertGreater(evaluate_coefficients([1.1,.01],[101])[0],1)
        self.assertLess(evaluate_coefficients([1,1],[300])[0],0)
    def test_invalid_future(self):
        for c in ([100],[np.inf],[]):
            with self.assertRaises(ValueError):evaluate_coefficients([1,.01],c)
    def test_family_rejected_before_inputs(self):
        with self.assertRaises(ValueError):predict(dict(family='A'),dict(physical_family='B'),[101])

if __name__=='__main__':unittest.main()
