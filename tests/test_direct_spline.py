import unittest
import numpy as np
from battery_smetimes.direct_spline import time_basis,fit,predict
from tests.test_conditional_drift import cell
from battery_smetimes.direct_full_spline import fit as full_fit,predict as full_predict


class DirectSplineTest(unittest.TestCase):
    def test_full_feature_training_offset_preserves_inference(self):
        cells={}
        for k,v in [('a',1),('b',2)]:
            c=cell(v);c.update(cycle=np.array([101,110,120,130]),truth=np.array([1,.98,.96,.94]));cells[k]=c
        b=full_fit(cells,['a','b'],10);c=cell(3);p=full_predict(b,c,[110,120])
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(b['scaler'].var_).all())
        self.assertTrue((b['scaler'].var_>=0).all())
        poison=dict(c,early_cycle=np.r_[c['early_cycle'],101],early_raw=np.vstack([c['early_raw'],[[np.inf,np.inf]]]),truth=np.inf)
        np.testing.assert_array_equal(p,full_predict(b,poison,[110,120]))

    def test_fixed_source_knots(self):
        x=time_basis([125,150,200]);self.assertEqual(x.shape,(3,8))
        np.testing.assert_allclose(x[:,2],[0,.25**2,.75**2])
        with self.assertRaises(ValueError):time_basis([100])

    def test_train_only_family_gate_and_future_poison(self):
        cells={}
        for k,v in [('a',1),('b',2)]:
            c=cell(v);c.update(cycle=np.array([101,110,120,130]),truth=np.array([1,.98,.96,.94]));cells[k]=c
        b=fit(cells,['a','b'],2,10);c=cell(3);p=predict(b,c,[110,120])
        poison=dict(c,early_cycle=np.r_[c['early_cycle'],101],early_raw=np.vstack([c['early_raw'],[[np.inf,np.inf]]]),truth=np.inf,cycle_life=np.inf)
        np.testing.assert_array_equal(p,predict(b,poison,[110,120]))
        with self.assertRaises(ValueError):predict(b,dict(c,physical_family='other'),[110])


if __name__=='__main__':unittest.main()
