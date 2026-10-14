import unittest
import numpy as np
from battery_smetimes.pooled_anchor import early,supervision


class PooledAnchorTests(unittest.TestCase):
    def cell(self):
        return dict(cycle=np.arange(1.,101.),raw=np.ones((100,44)),label_cycle=np.arange(1.,101.),soh=np.linspace(1,.9,100))

    def test_join_anchor_and_future_exclusion(self):
        c=self.cell();cycles,x,y,a=supervision(c);self.assertEqual(len(y),100);self.assertAlmostEqual(a,np.median(c['soh'][-3:]))
        poison=dict(c,cycle=np.r_[c['cycle'],101],raw=np.vstack([c['raw'],np.full((1,44),np.inf)]))
        for actual,expected in zip(early(poison),early(c)):np.testing.assert_array_equal(actual,expected)

    def test_missing_labels_and_width(self):
        c=self.cell();c['soh'][:4]=np.nan;cycle,_,y,_=supervision(c);self.assertEqual(cycle[0],5);self.assertEqual(len(y),96)
        with self.assertRaises(ValueError):early(dict(c,raw=c['raw'][:,:43]))
        with self.assertRaises(ValueError):supervision(dict(c,soh=np.full(100,np.nan)))


if __name__=='__main__':unittest.main()
