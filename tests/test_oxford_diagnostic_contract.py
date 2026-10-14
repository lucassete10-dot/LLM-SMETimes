import unittest
import numpy as np
from scripts.train_oxford_diagnostic_forecast import elapsed,diagnostic_features,matrix

class OxfordContractTests(unittest.TestCase):
    def record(self):
        t=np.arange(1001,dtype=float)
        return dict(t=t,v=4.1-t*.0003,T=40+t*.001,q=-t*.205)

    def test_q_not_an_input(self):
        r=self.record();p=diagnostic_features(r)
        r['q']=np.array([np.inf])
        np.testing.assert_array_equal(p,diagnostic_features(r))

    def test_signal_after_cutoff_not_an_input(self):
        r=self.record();p=diagnostic_features(r)
        r['v'][901:]=np.inf;r['T'][901:]=np.inf
        np.testing.assert_array_equal(p,diagnostic_features(r))

    def test_time_units(self):
        seconds=np.arange(1001,dtype=float)
        converted,unit=elapsed(735954.+seconds/86400.)
        np.testing.assert_allclose(converted,seconds,atol=.00002,rtol=0)
        self.assertIn('serial days',unit)
        np.testing.assert_array_equal(elapsed(seconds)[0],seconds)
        with self.assertRaises(ValueError):elapsed(np.array([0.,1.,1.]))

    def test_requested_cycle_does_not_require_label_or_eol(self):
        cell=dict(prefix=np.ones(68),truth=np.array([1.,.8]),cycle=np.array([200,5000]))
        expected=matrix(cell,np.array([200,300]))
        del cell['truth'];del cell['cycle']
        np.testing.assert_array_equal(matrix(cell,np.array([200,300])),expected)

if __name__=='__main__':unittest.main()
