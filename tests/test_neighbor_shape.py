import unittest
import numpy as np
from battery_smetimes.neighbor_shape import mix


class ShapeMixTest(unittest.TestCase):
    def test_preserves_predicted_anchor(self):
        self.assertAlmostEqual(mix([.95,.9],[1,.8],.95,1,.25)[0],.95)

    def test_reference_level_translation_invariant(self):
        np.testing.assert_allclose(mix([.95,.9],[1,.8],.95,1,.5),mix([.95,.9],[1.2,1],.95,1.2,.5),rtol=0,atol=1e-12)

    def test_zero_weight_preserves_primary(self):
        np.testing.assert_array_equal(mix([.95,.9],[1,.8],.95,1,0),[.95,.9])

    def test_anchor_cannot_change_same_weight_degradation(self):
        p=np.array([.95,.93,.9]);q=np.array([1,.92,.8]);weight=.25
        raw=(1-weight)*p+weight*q
        shape=mix(p,q,.95,1,weight)
        self.assertAlmostEqual(raw[0]-raw[-1],shape[0]-shape[-1])

    def test_rejects_invalid_data(self):
        for args in [([1],[np.nan],1,1,.5),([1],[1,2],1,1,.5),([1],[1],1,1,2)]:
            with self.assertRaises(ValueError):mix(*args)


if __name__=='__main__':unittest.main()
