import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from tests.test_conditional_drift import cell
from battery_smetimes.parameter_pls import fit,predict


class PLSTest(unittest.TestCase):
    def test_fit_membership_and_future_invariance(self):
        cells={k:cell(v) for k,v in [('a',1),('b',2),('unused',999)]}
        rows=[dict(cell_id=k,anchor=1.,rate=.01*v,acceleration=.002*v,knee_h=v) for k,v in [('a',1),('b',2)]]
        with threadpool_limits(limits=2):
            b=fit(cells,['a','b'],4,True,1,rows);q=cell(3);x=np.arange(101,200);p=predict(b,q,x)
            self.assertEqual(b['model'].n_components,1);self.assertEqual(b['train_ids'],['a','b'])
            self.assertTrue((np.diff(p)<=1e-12).all())
            poison=dict(q,early_cycle=np.r_[q['early_cycle'],101],early_raw=np.vstack([q['early_raw'],[[np.inf,np.inf]]]),truth=np.inf)
            np.testing.assert_array_equal(p,predict(b,poison,x))
            with self.assertRaises(ValueError):predict(b,dict(q,physical_family='different'),x)
            with self.assertRaises(ValueError):fit(cells,['a','b'],2,True,1,rows[::-1])


if __name__=='__main__':unittest.main()
