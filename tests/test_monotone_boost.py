import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.monotone_boost import fit,predict,inputs
from tests.test_conditional_drift import cell


class BoostTest(unittest.TestCase):
    def test_matched_inputs_and_constraints_only(self):
        cells={}
        for k,v in [('a',1),('b',2)]:
            c=cell(v);c.update(cycle=np.arange(101,161),truth=np.linspace(1,.8,60));cells[k]=c
        with threadpool_limits(limits=2):
            off=fit(cells,['a','b'],1,False,1);on=fit(cells,['a','b'],1,True,1)
            np.testing.assert_array_equal(off['imputer'].statistics_,on['imputer'].statistics_)
            x=inputs(on['imputer'],cell(3),[101,150,250])
            np.testing.assert_array_equal(x[:,:-3],np.tile(x[0,:-3],(3,1)))
            self.assertTrue((np.diff(x[:,-3:],axis=0)>0).all())
            np.testing.assert_array_equal(off['model'].monotonic_cst,np.zeros(x.shape[1]))
            with self.assertRaises(ValueError):predict(on,cell(3),[100])

    def test_monotone_training_only_future_invariance(self):
        cells={}
        for k,v in [('a',1),('b',2),('unused',999)]:
            c=cell(v);c.update(cycle=np.arange(101,161),truth=np.linspace(1,.8,60));cells[k]=c
        with threadpool_limits(limits=2):
            b=fit(cells,['a','b'],1,True,1);c=cell(3);p=predict(b,c,np.arange(101,200))
            self.assertTrue((np.diff(p)<=1e-12).all());self.assertEqual(b['train_ids'],['a','b'])
            self.assertFalse(b['model'].early_stopping)
            np.testing.assert_array_equal(b['model'].monotonic_cst[-3:],[-1,-1,-1])
            poison=dict(c,early_cycle=np.r_[c['early_cycle'],101],early_raw=np.vstack([c['early_raw'],[[np.inf,np.inf]]]),truth=np.inf)
            np.testing.assert_array_equal(p,predict(b,poison,np.arange(101,200)))
            with self.assertRaises(ValueError):predict(b,dict(c,physical_family='other'),[110])


if __name__=='__main__':unittest.main()
