import unittest
import numpy as np
from scripts.diagnose_mit_training_horizon_support import support


class SupportTest(unittest.TestCase):
    def cell(self,c,y):return dict(label_cycle=np.asarray(c),soh=np.asarray(y))

    def test_no_terminal_carry_forward(self):
        cells={'short':self.cell([101,200],[1,.8]),'long':self.cell([101,300],[1,.9])}
        r=support(cells,list(cells),300)
        self.assertEqual(r['cells_observed_to_horizon'],1)
        self.assertEqual(r['actual_label_count'],1)
        self.assertAlmostEqual(r['actual_soh_mean'],.9)
        self.assertAlmostEqual(r['observed_to_horizon_fraction'],.5)

    def test_gap_does_not_invent_observation(self):
        r=support({'x':self.cell([101,400],[1,.8])},['x'],250)
        self.assertEqual(r['cells_observed_to_horizon'],1)
        self.assertEqual(r['actual_label_count'],0)
        self.assertIsNone(r['actual_soh_mean'])

    def test_zero_support_not_zero_health(self):
        r=support({'x':self.cell([101,200],[1,.8])},['x'],300)
        self.assertEqual(r['cells_observed_to_horizon'],0)
        self.assertIsNone(r['actual_soh_median'])
