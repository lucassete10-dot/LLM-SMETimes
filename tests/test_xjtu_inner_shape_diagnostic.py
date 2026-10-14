import unittest
import numpy as np
from scripts.diagnose_xjtu_segment_inner_shape import decomposition

class DecompositionTests(unittest.TestCase):
    def test_offset_only(self):
        c=np.arange(101,201); y=1-(c-100)*.001
        r=decomposition(c,y,y+.02)
        self.assertAlmostEqual(r['start_signed_pp'],2)
        self.assertAlmostEqual(r['endpoint_signed_pp'],2)
        self.assertAlmostEqual(r['drop_abs_pp'],0)
        self.assertFalse(r['late_underdecline'])

    def test_flat_is_severe_not_just_offset(self):
        c=np.arange(101,201);y=1-(c-100)*.001
        r=decomposition(c,y,np.full(len(c),y[0]))
        self.assertTrue(r['severe_underdecline'])
        self.assertTrue(r['late_underdecline'])
        self.assertAlmostEqual(r['change_signed_pp'],9.9)

    def test_actual_sparse_cycles(self):
        c=np.array([101,130,150,200]);y=1-c*.001
        r=decomposition(c,y,y)
        self.assertAlmostEqual(r['late_true_decline_pp_per_100_cycles'],10)
        self.assertAlmostEqual(r['mape_percent'],0)

    def test_bad_order(self):
        with self.assertRaises(ValueError):decomposition([102,101],[1,.9],[1,.9])

if __name__=='__main__': unittest.main()
