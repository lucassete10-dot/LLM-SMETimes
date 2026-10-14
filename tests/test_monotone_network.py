import unittest
import numpy as np
import torch
from battery_smetimes.monotone_network import CurveNet,fit,predict
from tests.test_conditional_drift import cell


class MonotoneTest(unittest.TestCase):
    def test_positive_rates_monotone_curve(self):
        torch.manual_seed(1);model=CurveNet(3,16);x=torch.randn(5,3)
        params=model.parameters_for(x);self.assertTrue(torch.all(params[:,1:]>=0))
        h=torch.linspace(0,20,50);p=model(x,torch.zeros(50,dtype=torch.long),h)
        self.assertTrue(torch.all(torch.diff(p)<=0))

    def test_label_free_poison_and_family_gate(self):
        cells={}
        for key,value in [('a',1),('b',2)]:
            c=cell(value);c.update(cycle=np.array([101,110,120]),truth=np.array([1,.98,.95]));cells[key]=c
        b=fit(cells,['a','b'],16,.1,1,epochs=2);c=cell(3);p=predict(b,c,[110,120])
        poison=dict(c,early_cycle=np.r_[c['early_cycle'],101],early_raw=np.vstack([c['early_raw'],[[np.inf,np.inf]]]),truth=np.inf,cycle_life=np.inf)
        np.testing.assert_array_equal(p,predict(b,poison,[110,120]))
        with self.assertRaises(ValueError):predict(b,dict(c,physical_family='other'),[110])


if __name__=='__main__':unittest.main()
