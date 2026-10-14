import unittest
from unittest.mock import patch
import numpy as np
from battery_smetimes.decoupled_anchor import make,predict


class DecoupledTests(unittest.TestCase):
    def test_constant_shift_preserves_dynamics(self):
        p=dict(family='x',train_ids=['a'],seed=1);r=dict(p,learned_anchor=False)
        b=make(p,r);cycle=np.array([101.,200.,300.])
        with patch('battery_smetimes.decoupled_anchor.anchor',return_value=.9),patch('battery_smetimes.decoupled_anchor.fade',side_effect=lambda b,c,x:1-np.asarray(x)/1000):
            out=predict(b,dict(physical_family='x'),cycle)
        np.testing.assert_allclose(out,.9-(cycle-101)/1000,atol=1e-15,rtol=0)
        with self.assertRaises(ValueError):make(p,dict(r,train_ids=['b']))
        with self.assertRaises(ValueError):predict(b,dict(physical_family='wrong'),cycle)


if __name__=='__main__':unittest.main()
