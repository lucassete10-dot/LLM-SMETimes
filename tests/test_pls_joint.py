import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from tests.test_conditional_drift import cell
from battery_smetimes.parameter_pls import fit as pls_fit,predict as pls_predict
from battery_smetimes.pls_joint import fit,predict


class JointTest(unittest.TestCase):
    def test_initializer_matches_and_future_isolation(self):
        cells={}
        for k,v in [('a',1),('b',2)]:
            c=cell(v);c.update(cycle=np.arange(101,161),truth=np.linspace(1,.8,60));cells[k]=c
        rows=[dict(cell_id=k,anchor=1.,rate=.01*v,acceleration=.002*v,knee_h=v) for k,v in [('a',1),('b',2)]]
        with threadpool_limits(limits=2):
            encoder=pls_fit(cells,['a','b'],2,True,1,rows);b=fit(cells,['a','b'],encoder,.01,1,epochs=0);q=cell(3);x=np.arange(101,200)
            np.testing.assert_allclose(predict(b,q,x),pls_predict(encoder,q,x),atol=1e-10,rtol=0)
            trained=fit(cells,['a','b'],encoder,.01,1,epochs=2);p=predict(trained,q,x)
            self.assertTrue((np.diff(p)<=1e-12).all());self.assertEqual(len(trained['history']),2)
            poison=dict(q,early_cycle=np.r_[q['early_cycle'],101],early_raw=np.vstack([q['early_raw'],[[np.inf,np.inf]]]),truth=np.inf)
            np.testing.assert_array_equal(p,predict(trained,poison,x))
            with self.assertRaises(ValueError):predict(trained,dict(q,physical_family='different'),x)


if __name__=='__main__':unittest.main()
