import numpy as np
import unittest
from battery_smetimes.mit_completed_signal import cycle_features


def signals():
    t=np.linspace(0,15,901)
    return t,np.full(901,-4.4),np.linspace(3.6,2,901),np.full(901,30.)


class CompletedSignalTests(unittest.TestCase):
    def test_current_integral_and_window_conservation(self):
        x=cycle_features(*signals())
        self.assertEqual(x.shape,(25,))
        np.testing.assert_allclose([x[0],x[1],x[9::2].sum(),x[10::2].sum()],[900,1.1,900,1.1])


    def test_reject_time_reversal(self):
        t,i,v,temp=signals();t[100]=t[99]-.1
        with self.assertRaisesRegex(ValueError,'nonmonotone'):
            cycle_features(t,i,v,temp)


    def test_reject_nonfinite_and_length(self):
        t,i,v,temp=signals();v[30]=np.inf
        with self.assertRaisesRegex(ValueError,'nonfinite'):cycle_features(t,i,v,temp)
        with self.assertRaisesRegex(ValueError,'length'):cycle_features(t,i,v,temp[:-1])


if __name__=='__main__':unittest.main()
