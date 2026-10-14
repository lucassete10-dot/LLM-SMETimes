import unittest
import numpy as np
from battery_smetimes.nasa_cycle_calibration import features,supervision,early_health


class CalibrationTests(unittest.TestCase):
    def cell(self):
        return dict(cycle=np.arange(1,103),raw=np.arange(102*44,dtype=float).reshape(102,44),metadata=dict(discharge_cutoff_v=2.7),physical_family='NASA',label_cycle=np.arange(1,103),soh=np.linspace(1,.8,102))

    def test_future_signals_do_not_enter_features(self):
        c=self.cell();a=features(c,'absolute')[1];c['raw'][100:]=np.inf;c['soh']=np.inf
        np.testing.assert_array_equal(a,features(c,'absolute')[1])

    def test_delta_reference_is_first_five_and_condition_retained(self):
        c=self.cell();_,x=features(c,'delta');np.testing.assert_array_equal(x[2,:6],np.zeros(6));np.testing.assert_array_equal(x[:,-1],np.full(100,2.7))

    def test_supervision_only_joins_early_train_labels(self):
        c=self.cell();c['soh'][100:]=999;d=self.cell();d['soh'][::2]=np.nan;x,y,w,m=supervision({'a':c,'b':d},['a','b'],'absolute')
        self.assertTrue(np.all(y<2));self.assertTrue(all(r['cycle']<=100 for r in m));self.assertAlmostEqual(w[:100].sum(),w[100:].sum());self.assertEqual(len(x),150)

    def test_different_family_rejected_before_prediction(self):
        with self.assertRaises(ValueError):early_health({'family':'other'},self.cell())


if __name__=='__main__':unittest.main()
