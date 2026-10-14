import unittest
import numpy as np
import pandas as pd
from scripts.prepare_zhu2022_nca import signal_features,completed_groups,capacity_label

class NCAInputs(unittest.TestCase):
    def frame(self):
        t=np.arange(0.,1101.,10.)
        return pd.DataFrame({'time/s':t,'Ecell/V':4.1-t*.0002,'<I>/mA':-3500.+t*.0001,
            'Q discharge/mA.h':t*.9,'cycle number':np.ones(len(t))})

    def test_capacity_is_not_input(self):
        f=self.frame();expected=signal_features(f);f['Q discharge/mA.h']=np.inf
        np.testing.assert_array_equal(expected,signal_features(f))

    def test_after900_voltage_current_poison_has_no_effect(self):
        f=self.frame();expected=signal_features(f)
        f.loc[f['time/s']>900,['Ecell/V','<I>/mA']]=np.inf
        np.testing.assert_array_equal(expected,signal_features(f))

    def test_chunk_boundary_keeps_whole_cycles(self):
        f=self.frame().iloc[:6].copy();f['cycle number']=[1,1,2,2,3,3]
        rows=list(completed_groups([f.iloc[:3],f.iloc[3:]]))
        self.assertEqual([int(n) for n,g in rows],[1,2,3]);self.assertEqual([len(g) for n,g in rows],[2,2,2])

    def test_counter_restart_not_silently_joined(self):
        f=self.frame().iloc[:3].copy();f['cycle number']=[1,2,1]
        with self.assertRaises(ValueError):list(completed_groups([f]))

    def label_frame(self):
        t=np.r_[0.,np.arange(10.,911.,10.)]
        q=np.r_[0.,np.arange(0.,901.,10.)*3500/3600]
        return pd.DataFrame({'time/s':t,'Ecell/V':np.r_[4.2,np.linspace(4.05,2.65,91)],
            '<I>/mA':np.r_[0.,np.full(91,-3500.)],'cycle number':np.ones(92),'Q discharge/mA.h':q})

    def test_low_capacity_complete_discharge_is_retained(self):
        self.assertAlmostEqual(capacity_label(self.label_frame()),875.)

    def test_partial_low_soc_record_not_full_capacity(self):
        f=self.label_frame();f.loc[1:,'Ecell/V']=np.linspace(2.9,2.65,91)
        with self.assertRaises(ValueError):capacity_label(f)

    def test_bad_q_rejected_by_current_not_prediction_error(self):
        f=self.label_frame();f['Q discharge/mA.h']*=.1
        with self.assertRaises(ValueError):capacity_label(f)

if __name__=='__main__':unittest.main()
