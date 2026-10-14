import unittest
import numpy as np
from battery_smetimes.temporal_features import temporal_vector


class TemporalFeaturesTest(unittest.TestCase):
    def test_fixed_windows_and_future_invariance(self):
        c=np.arange(1,101);raw=(c/100)[:,None]
        cell=dict(early_cycle=c,early_raw=raw)
        v=temporal_vector(cell)
        self.assertEqual(v.shape,(11,))
        np.testing.assert_allclose(v[-3:],[1,1,0],atol=1e-12)
        other=dict(cell,early_cycle=np.r_[c,101,10000],early_raw=np.vstack([raw,[[np.inf],[np.inf]]]),truth=np.inf,cycle_life=9999)
        np.testing.assert_array_equal(v,temporal_vector(other))

    def test_missing_window_stays_missing(self):
        v=temporal_vector(dict(early_cycle=np.array([1,2,3]),early_raw=np.ones((3,2))))
        self.assertTrue(np.isnan(v[4:16]).all())


if __name__=='__main__':unittest.main()
