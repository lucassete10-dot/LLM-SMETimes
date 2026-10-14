import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from tests.test_conditional_drift import cell
from battery_smetimes.decorrelated_forest import fit,predict


class ForestTest(unittest.TestCase):
    def test_training_membership_and_future_invariance(self):
        cells={}
        for k,v in [('a',1),('b',2),('unused',999)]:
            c=cell(v);c.update(cycle=np.arange(101,161),truth=np.linspace(1,.8,60));cells[k]=c
        with threadpool_limits(limits=2):
            b=fit(cells,['a','b'],.25,True,1);q=cell(3);x=np.arange(101,200);p=predict(b,q,x)
            self.assertEqual(b['train_ids'],['a','b']);self.assertEqual(b['model'].max_features,.25)
            self.assertEqual(b['model'].n_estimators,128)
            poison=dict(q,early_cycle=np.r_[q['early_cycle'],101],early_raw=np.vstack([q['early_raw'],[[np.inf,np.inf]]]),truth=np.inf,cycle_life=np.inf)
            np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
            with self.assertRaises(ValueError):predict(b,dict(q,physical_family='different'),x)


if __name__=='__main__':unittest.main()
