import unittest
import numpy as np

from battery_smetimes.curve_fidelity import curve_metrics, selection_score, shape_parameters


class CurveFidelityTests(unittest.TestCase):
    def test_flat_curve_is_not_a_joint_pass(self):
        x = np.arange(101, 148)
        y = 1.015-.135*np.linspace(0, 1, len(x))**2
        p = np.linspace(1.024, 1.011, len(x))
        m = curve_metrics(x, y, p)
        self.assertLess(m['mape_percent'], 6)
        self.assertTrue(m['severe_flat_failure'])
        self.assertFalse(m['joint_pass'])
        self.assertGreater(m['drop_error_pp'], 10)

    def test_shape_learns_acceleration_without_life_scaling(self):
        x = np.arange(101, 401)
        h = (x-100)/100
        y = 1.02-.02*h-.01*h*h
        coefficients = shape_parameters(x, y)
        np.testing.assert_allclose(coefficients, [1.02, .02, .01], atol=1e-4)
        self.assertGreater(coefficients[0], 1)

    def test_perfect_curve_has_zero_selection_score(self):
        x = np.arange(101, 201); y = np.linspace(1, .85, len(x))
        row = curve_metrics(x, y, y)
        self.assertTrue(row['joint_pass'])
        self.assertEqual(selection_score([row]), 0)


if __name__ == '__main__':
    unittest.main()
