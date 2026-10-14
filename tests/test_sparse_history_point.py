import unittest
import numpy as np
from battery_smetimes import sparse_history_point as sparse,continuous_length as original


class SparsePrefixTests(unittest.TestCase):
    def cell(self,n):
        return dict(early_cycle=np.arange(1,n+1,dtype=float),early_raw=np.arange(n*4,dtype=float).reshape(n,4),conditions=np.array([30.]))

    def test_three_observations_exactly_preserve_original(self):
        cell=self.cell(5)
        np.testing.assert_equal(sparse.features(cell,3),original.features(cell,3))

    def test_single_observation_does_not_fabricate_trend(self):
        cell=self.cell(1);f=sparse.features(cell,10)
        np.testing.assert_equal(f[:4],cell['early_raw'][0])
        self.assertTrue(np.isnan(f[12:16]).all())
        self.assertAlmostEqual(sum(f[-5:-1])*10,1)

    def test_future_poison_has_no_effect(self):
        cell=self.cell(5);expected=sparse.features(cell,2)
        cell['early_raw'][2:]=np.inf
        np.testing.assert_equal(sparse.features(cell,2),expected)

    def test_no_observations_rejected(self):
        with self.assertRaises(ValueError):sparse.features(self.cell(3),0)


if __name__=='__main__':unittest.main()
