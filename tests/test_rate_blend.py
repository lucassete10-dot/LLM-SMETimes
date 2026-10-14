import unittest
from unittest.mock import patch
import numpy as np
from battery_smetimes.rate_blend import make,predict


class RateBlendTests(unittest.TestCase):
    def test_global_weight_and_family(self):
        primary=dict(family='MIT_LFP',train_ids=['a','b']);rate=dict(primary)
        with patch('battery_smetimes.rate_blend.primary_predict',return_value=np.array([1.,.9,.8])),patch('battery_smetimes.rate_blend.rate_predict',return_value=np.array([.99,.85,.7])):
            p=predict(make(primary,rate,.25),dict(physical_family='MIT_LFP'),np.arange(3))
            np.testing.assert_allclose(p,[.9975,.8875,.775],atol=1e-12,rtol=0)
            with self.assertRaises(ValueError):predict(make(primary,rate,.25),dict(physical_family='NASA'),np.arange(3))

    def test_bad_components_and_weights(self):
        primary=dict(family='MIT_LFP',train_ids=['a','b'])
        for other in [dict(family='NASA',train_ids=['a','b']),dict(family='MIT_LFP',train_ids=['b','a'])]:
            with self.assertRaises(ValueError):make(primary,other,.5)
        for w in [-.1,1.1,np.nan,np.inf]:
            with self.assertRaises(ValueError):make(primary,dict(primary),w)


if __name__=='__main__':unittest.main()
