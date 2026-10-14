import unittest
import numpy as np
from battery_smetimes.xjtu_full_signal import cycle_features


class FullSignalTests(unittest.TestCase):
    def row(self):
        return dict(relative_time_min=[0,1,2,0,1,2,0,1,2,0,1,2],
            voltage_V=[3.5]*12,current_A=[2,2,2,0,0,0,-2,-2,-2,0,0,0],temperature_C=[25]*12)

    def test_integral_and_stage_separation(self):
        x=cycle_features(self.row())
        self.assertEqual(x.shape,(44,))
        np.testing.assert_allclose(x[:7],[120,240/3600,2,3.5,0,25,25])
        np.testing.assert_allclose(x[22:29],x[:7])

    def test_capacity_not_read(self):
        r=self.row();expected=cycle_features(r)
        r.update(capacity=np.inf,soh=np.inf,cycle_life=np.inf)
        np.testing.assert_equal(expected,cycle_features(r))

    def test_nonmonotone_rejected(self):
        r=self.row();r['relative_time_min'][2]=.5
        with self.assertRaises(ValueError):cycle_features(r)

    def test_sign_rejected(self):
        r=self.row();r['current_A'][6:9]=[2]*3
        with self.assertRaises(ValueError):cycle_features(r)


if __name__=='__main__':unittest.main()
