import unittest
from unittest.mock import patch
import numpy as np
from scripts.train_hgb_rate_smoke import supervision

class RateSupervisionTest(unittest.TestCase):
    def test_anchor_rate_and_order(self):
        cycles={'a':np.array([101.,150.,200.]),'b':np.array([110.,210.])}
        truth={'a':np.array([.99,.96,.90]),'b':np.array([1.,.85])}
        y=np.r_[truth['a'],truth['b']];x=np.arange(10).reshape(5,2);w=np.ones(5)
        with patch('scripts.train_hgb_rate_smoke.train_arrays',return_value=(x,y,w)),patch('scripts.train_hgb_rate_smoke.labels',side_effect=lambda c:(cycles[c],truth[c])):
            xx,r,ww=supervision({'a':'a','b':'b'},['a','b'],[1.02,1.01])
        np.testing.assert_array_equal(xx,x);np.testing.assert_array_equal(ww,w)
        anchors=np.array([1.02]*3+[1.01]*2);coords=np.r_[cycles['a'],cycles['b']]
        np.testing.assert_allclose(anchors-(coords/100)*r,y,atol=1e-14,rtol=0)

if __name__=='__main__':unittest.main()
