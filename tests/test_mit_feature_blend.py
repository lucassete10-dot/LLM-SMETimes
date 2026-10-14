import unittest
from unittest.mock import patch
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from battery_smetimes.mit_feature_blend import predict,estimator_components

class BlendTests(unittest.TestCase):
    def test_uniform_mix(self):
        b=dict(family='A',new_weight=.5,old={},new={});q=dict(physical_family='A')
        with patch('battery_smetimes.mit_feature_blend.legacy_predict',return_value=np.array([1,.9])),patch('battery_smetimes.mit_feature_blend.feature_predict',return_value=np.array([.98,.8])):
            np.testing.assert_allclose(predict(b,q,[101,200]),[.99,.85])
    def test_family_and_weight_gate(self):
        with self.assertRaises(ValueError):predict(dict(family='A'),dict(physical_family='B'),[101])
        with self.assertRaises(ValueError):predict(dict(family='A',new_weight=.3),dict(physical_family='A'),[101])
    def test_predictor_units_not_trees_or_imputers(self):
        m=ExtraTreesRegressor(n_estimators=128)
        self.assertEqual(estimator_components(dict(model=m,alias=m,imputer=SimpleImputer(),local=[dict(model=ExtraTreesRegressor())])),2)

if __name__=='__main__':unittest.main()
