import unittest

import numpy as np

from evaluate_condition_features_v2 import condition_vector


class ConditionFeatureTests(unittest.TestCase):
    def test_known_and_unknown_are_explicit(self):
        vector = condition_vector({'charge_rate_c': 2., 'ambient_temperature_c': None,
                                   'charge_rate_steps_c': [3., 4.]})
        self.assertEqual(vector.shape, (14,))
        self.assertEqual(vector[0], 2.)
        self.assertTrue(np.isnan(vector[1]))
        self.assertEqual(vector[4], 3.)
        self.assertEqual(vector[5], 4.)
        self.assertEqual(vector[7], 0.)
        self.assertEqual(vector[8], 1.)
        self.assertEqual(vector[13], 1.)


if __name__ == '__main__':
    unittest.main()
