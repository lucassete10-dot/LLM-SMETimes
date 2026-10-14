import unittest
from unittest.mock import patch
import numpy as np
from battery_smetimes.nasa_slope_blend import predict


class SlopeBlendTests(unittest.TestCase):
    def test_anchor_equal_but_drop_changes(self):
        def base(bundle,query,origin,c):return 1-.001*(np.asarray(c)-101)
        def cal(bundle,query,c):return .9-.002*(np.asarray(c)-101)
        with patch('battery_smetimes.nasa_slope_blend.predict_baseline',side_effect=base),patch('battery_smetimes.nasa_slope_blend.calibrated',side_effect=cal):
            p=predict({},{},{},[101,201],.5);np.testing.assert_allclose(p,[.9,.75],rtol=0,atol=1e-12)

    def test_endpoints_and_bad_weights(self):
        for w in (-1,2,np.nan,np.inf):
            with self.assertRaises(ValueError):predict({},{},{},[101,201],w)
        with patch('battery_smetimes.nasa_slope_blend.predict_baseline',return_value=np.array([1.])),patch('battery_smetimes.nasa_slope_blend.calibrated',return_value=np.array([.8])):
            np.testing.assert_allclose(predict({},{},{},[101],0),[.8]);np.testing.assert_allclose(predict({},{},{},[101],1),[.8])


if __name__=='__main__':unittest.main()
