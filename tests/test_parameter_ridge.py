import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from tests.test_conditional_drift import cell
from battery_smetimes.parameter_ridge import transformed,parameters_from,fit,predict


class ParameterTest(unittest.TestCase):
    def test_transforms(self):
        rows=[dict(anchor=1.,rate=.01,acceleration=.002,knee_h=2.)]
        expected=np.array([[1.,.01,.002,2.]])
        for log in (False,True):np.testing.assert_allclose(parameters_from(transformed(rows,log),log),expected,atol=1e-12)

    def test_training_only_and_family_gate(self):
        cells={}
        for k,v in [('a',1),('b',2),('unused',999)]:
            c=cell(v);c.update(cycle=np.arange(101,161),truth=np.linspace(1,.8,60));cells[k]=c
        with threadpool_limits(limits=2):
            b=fit(cells,['a','b'],10,True,1);q=cell(3);x=np.arange(101,200);p=predict(b,q,x)
            self.assertEqual([r['cell_id'] for r in b['training_parameter_fits']],['a','b'])
            self.assertTrue((np.diff(p)<=1e-12).all())
            poison=dict(q,early_cycle=np.r_[q['early_cycle'],101],early_raw=np.vstack([q['early_raw'],[[np.inf,np.inf]]]),truth=np.inf)
            np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
            with self.assertRaises(ValueError):predict(b,dict(q,physical_family='different'),x)


if __name__=='__main__':unittest.main()
