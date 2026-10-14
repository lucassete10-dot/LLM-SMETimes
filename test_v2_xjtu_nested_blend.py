import unittest

import numpy as np

from evaluate_xjtu_nested_blend_v2 import blend


class NestedBlendTests(unittest.TestCase):
    def test_weight_endpoints_and_midpoint(self):
        full = {'cell': np.array([.8, .9])}
        raw = {'cell': np.array([1., .7])}
        np.testing.assert_allclose(blend(full, raw, 0)['cell'], [.8, .9])
        np.testing.assert_allclose(blend(full, raw, 1)['cell'], [1., .7])
        np.testing.assert_allclose(blend(full, raw, .5)['cell'], [.9, .8])
        with self.assertRaises(ValueError):
            blend(full, raw, 1.1)


if __name__ == '__main__':
    unittest.main()
