import unittest
import numpy as np
from scripts.train_curve_fidelity import load_subset
from battery_smetimes.xjtu_robust_trend import prefix,inputs
from battery_smetimes.xjtu_segment_features import prefix as original,columns


class RobustTrendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell=load_subset(['b1_2C_battery-3'])['b1_2C_battery-3'];cls.parameters=dict(block='VIT',context='segments')

    def test_other_features_unchanged(self):
        p=prefix(self.cell,self.parameters);q=original(self.cell,self.parameters);n=len(columns('VIT'))
        np.testing.assert_array_equal(p[:3*n],q[:3*n]);np.testing.assert_array_equal(p[4*n:],q[4*n:])

    def test_future_and_labels_unused(self):
        cell=self.cell;poison=dict(cell,soh=np.inf,label_cycle=np.inf,cycle=np.r_[cell['cycle'],10000.],raw=np.vstack([cell['raw'],np.full((1,cell['raw'].shape[1]),np.inf)]))
        np.testing.assert_array_equal(prefix(cell,self.parameters),prefix(poison,self.parameters))

    def test_invalid_request(self):
        for c in ([],[100],[np.inf],[[101]]):
            with self.assertRaises(ValueError):inputs(self.cell,c,self.parameters)


if __name__=='__main__':unittest.main()
