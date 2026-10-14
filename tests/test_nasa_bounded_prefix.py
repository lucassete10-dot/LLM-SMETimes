import unittest
from unittest.mock import patch
import numpy as np
from scripts.train_nasa_bounded_prefix import predict

class Identity:
    def transform(self,x):return x.copy()

class Capture:
    def predict(self,x):
        self.design=x.copy()
        return x[:,0]+x[:,-2]

class BoundedPrefixTests(unittest.TestCase):
    def bundle(self,limit):
        return dict(physical_family='test_family',registry={},imputer=Identity(),scaler=Identity(),
            model=Capture(),parameters=dict(prefix_z_limit=limit))

    def run_predict(self,bundle):
        with patch('scripts.train_nasa_bounded_prefix.require_same_registered_family'),patch(
            'scripts.train_nasa_bounded_prefix.signal_prefix',return_value=np.array([999.,-999.])):
            return predict(bundle,dict(physical_family='test_family'),100,np.array([200.,100100.]))

    def test_bounds_prefix_only_not_horizon_or_output(self):
        bundle=self.bundle(3.);p=self.run_predict(bundle);x=bundle['model'].design
        np.testing.assert_array_equal(x[:,:2],[[3.,-3.],[3.,-3.]])
        np.testing.assert_array_equal(x[:,-2],[1.,1000.])
        np.testing.assert_array_equal(p,[4.,1003.])

    def test_none_reproduces_unbounded_inputs(self):
        bundle=self.bundle(None);self.run_predict(bundle)
        np.testing.assert_array_equal(bundle['model'].design[:,:2],[[999.,-999.],[999.,-999.]])

if __name__=='__main__':unittest.main()
