"""Signal-only window integration and causal-prefix regression checks."""
import unittest
import numpy as np
from battery_smetimes.voltage_interval_features import cycle_features,features


class VoltageIntervalTests(unittest.TestCase):
    def test_constant_charge_integral(self):
        row=dict(time_in_s=[0,10,20],voltage_in_V=[3.35]*3,current_in_A=[2]*3)
        x=cycle_features(row)
        np.testing.assert_allclose(x[:7],[20,40/3600,2,0,3.35,0,1],atol=1e-12)
        self.assertEqual(x[7],0)
        self.assertTrue(np.isnan(x[9]))
        # Additional target fields have no effect on the explicitly whitelisted calculation.
        row.update(capacity=np.inf,soh=np.inf,cycle_life=np.inf)
        np.testing.assert_equal(cycle_features(row),x)

    def test_stage_boundary_is_not_bridged(self):
        row=dict(time_in_s=[0,10,20,1000,1010,1020],
                 voltage_in_V=[3.35]*6,current_in_A=[2,2,2,-2,-2,-2])
        x=cycle_features(row)
        self.assertEqual(x[0],20)
        self.assertAlmostEqual(x[1],40/3600)

    def test_constant_discharge_integral(self):
        x=cycle_features(dict(time_in_s=[0,10,20],voltage_in_V=[3.05]*3,current_in_A=[-2]*3))
        np.testing.assert_allclose(x[21:28],[20,40/3600,2,0,3.05,0,1],atol=1e-12)

    def test_bad_time_rejected(self):
        with self.assertRaises(ValueError):
            cycle_features(dict(time_in_s=[0,0,20],voltage_in_V=[3.35]*3,current_in_A=[2]*3))

    def test_future_signal_and_labels_do_not_change_prefix(self):
        c=np.arange(1,6,dtype=float)
        cell=dict(early_cycle=c,early_raw=np.ones((5,32)),conditions=np.array([]),
                  physics_cycle=c,physics_raw=np.ones((5,42)))
        actual=features(cell,3)
        cell['early_raw'][3:]=np.inf
        cell['physics_raw'][3:]=np.inf
        cell['truth']=np.inf
        np.testing.assert_equal(features(cell,3),actual)


if __name__=='__main__': unittest.main()
