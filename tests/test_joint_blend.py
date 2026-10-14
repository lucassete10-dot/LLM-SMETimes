import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from tests.test_conditional_drift import cell
from battery_smetimes.parameter_pls import fit as pls_fit
from battery_smetimes.pls_joint import fit as joint_fit,predict as joint_predict
from battery_smetimes.tail_boost import fit as tree_fit,predict as tree_predict
from battery_smetimes.joint_blend import make,predict


class BlendTest(unittest.TestCase):
    def test_global_weight_and_family_gate(self):
        cells={}
        for k,v in [('a',1),('b',2)]:
            c=cell(v);c.update(cycle=np.arange(101,161),truth=np.linspace(1,.8,60));cells[k]=c
        rows=[dict(cell_id=k,anchor=1.,rate=.01*v,acceleration=.002*v,knee_h=v) for k,v in [('a',1),('b',2)]]
        with threadpool_limits(limits=2):
            encoder=pls_fit(cells,['a','b'],2,True,1,rows);joint=joint_fit(cells,['a','b'],encoder,.01,1,epochs=0)
            tree=tree_fit(cells,['a','b'],dict(model='extra_trees',leaf=1,tail_weight=3,local=False),1)
            primary=dict(kind='tail_primary',mode='tail',inner=tree,family=tree['family'],train_ids=['a','b']);q=cell(3);x=np.arange(101,200)
            for w in (0,.25,1):np.testing.assert_allclose(predict(make(primary,joint,w),q,x),(1-w)*tree_predict(tree,q,x)+w*joint_predict(joint,q,x),atol=1e-12,rtol=0)
            with self.assertRaises(ValueError):make(primary,joint,1.1)
            with self.assertRaises(ValueError):make(dict(primary,train_ids=['a']),joint,.5)
            with self.assertRaises(ValueError):predict(make(primary,joint,.5),dict(q,physical_family='different'),x)


if __name__=='__main__':unittest.main()
