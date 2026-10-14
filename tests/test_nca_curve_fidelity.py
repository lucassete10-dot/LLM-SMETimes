import unittest
import numpy as np
from scripts.train_nca_curve_fidelity import fit,predict
from scripts.prepare_zhu2022_nca import prefix


def sample(offset=0):
    cycle=np.arange(1,101,dtype=float)
    raw=np.tile((4-.001*cycle+offset)[:,None],(1,32))
    conditions=np.array([25.,.5,1.,2.65,4.2])
    future=np.arange(101,151,dtype=float)
    return dict(early_cycle=cycle,early_raw=raw,conditions=conditions,
        prefix=prefix(cycle,raw,conditions),cycle=future,truth=1-.001*(future-100)+offset,
        physical_family='synthetic_NCA_family')


class NcaCurveFidelityTests(unittest.TestCase):
    def test_raw_replay_ignores_stored_prefix_labels_and_future(self):
        cells={str(i):sample(i*.002) for i in range(6)}
        b=fit(cells,list(cells),dict(local=True,trees=16,depth=5,leaf=1),20261005)
        self.assertEqual(len(b['local']),1)
        c=sample(.003);future=c['cycle'].copy();p=predict(b,c,future)
        c.pop('truth');c.pop('cycle');c['prefix'][:]=np.inf
        c['early_cycle']=np.r_[c['early_cycle'],101,10000]
        c['early_raw']=np.vstack([c['early_raw'],np.full((2,32),np.inf)])
        np.testing.assert_allclose(p,predict(b,c,future),atol=1e-12,rtol=0)
        c['physical_family']='different'
        with self.assertRaises(ValueError):predict(b,c,future)

    def test_small_condition_group_uses_global_fallback(self):
        cells={'a':sample(),'b':sample(.01)}
        b=fit(cells,list(cells),dict(local=True,trees=16,depth=5,leaf=1),20261005)
        self.assertEqual(b['local'],{})
        self.assertEqual(b['train_ids'],['a','b'])


if __name__=='__main__':unittest.main()
