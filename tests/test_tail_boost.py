import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import fit,predict,train_arrays
from tests.test_nca_curve_fidelity import sample


class TailBoostTests(unittest.TestCase):
    def test_tail_weight_preserves_equal_cell_contribution(self):
        cells={'a':sample(),'b':sample(.01)}
        _,_,w=train_arrays(cells,['a','b'],3)
        self.assertAlmostEqual(w[:50].sum(),w[50:].sum())
        self.assertAlmostEqual(w[49]/w[0],3)

    def test_boosting_disables_random_row_early_stopping_and_poison(self):
        cells={'a':sample(),'b':sample(.01)}
        with threadpool_limits(limits=2):
            b=fit(cells,['a','b'],dict(model='hist_gradient_boosting',max_leaves=15,tail_weight=3,local=False),20261005)
            self.assertFalse(b['model'].early_stopping)
            c=sample(.003);future=c['cycle'].copy();p=predict(b,c,future)
            c.pop('truth');c.pop('cycle');c['prefix'][:]=np.inf
            c['early_cycle']=np.r_[c['early_cycle'],101,10000]
            c['early_raw']=np.vstack([c['early_raw'],np.full((2,32),np.inf)])
            np.testing.assert_array_equal(p,predict(b,c,future))
            c['physical_family']='different'
            with self.assertRaises(ValueError):predict(b,c,future)


if __name__=='__main__':unittest.main()
