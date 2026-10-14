import unittest
import numpy as np
from battery_smetimes.relative_anchor import signals,training


class RelativeAnchorTests(unittest.TestCase):
    def cell(self):
        cycle=np.arange(1.,101.)
        return dict(cycle=cycle,raw=np.outer(cycle,np.arange(1.,45.))/100,label_cycle=cycle,soh=np.linspace(1,.9,100))

    def test_offset_and_future_invariance(self):
        c=self.cell();shift=dict(c,raw=c['raw']+np.arange(44))
        poison=dict(c,cycle=np.r_[c['cycle'],101],raw=np.vstack([c['raw'],np.full((1,44),np.inf)]),soh=np.inf)
        for compact in (False,True):
            cycle,x=signals(c,compact)
            np.testing.assert_allclose(x,signals(shift,compact)[1],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(cycle,signals(poison,compact)[0])
            np.testing.assert_array_equal(x,signals(poison,compact)[1])

    def test_join_and_cell_equal_weights(self):
        a=self.cell();b=self.cell();b['soh']=b['soh'].copy();b['soh'][:50]=np.nan
        x,y,w,m=training({'a':a,'b':b},['a','b'],True)
        self.assertEqual(x.shape,(150,6));self.assertEqual(m[1]['cycles'][0],51.)
        self.assertAlmostEqual(w[:100].sum(),w[100:].sum())
        np.testing.assert_array_equal(y,np.r_[a['soh'],b['soh'][50:]])


if __name__=='__main__':unittest.main()
