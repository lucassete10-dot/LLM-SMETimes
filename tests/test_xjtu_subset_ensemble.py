import unittest
import numpy as np
from battery_smetimes.xjtu_subset_ensemble import combine


class SubsetEnsembleTest(unittest.TestCase):
    def test_mean(self):np.testing.assert_array_equal(combine([[1,2],[3,4]],'mean'),[2,3])

    def test_median(self):np.testing.assert_array_equal(combine([[1,8],[7,2],[3,4],[5,6],[9,0]],'median'),[5,4])

    def test_full(self):np.testing.assert_array_equal(combine([[1,2]],'full'),[1,2])

    def test_invalid(self):
        for p,kind in [([], 'mean'),([[np.inf]],'mean'),([[1],[2]],'full'),([[1]],'other')]:
            with self.assertRaises(ValueError):combine(p,kind)


if __name__=='__main__':unittest.main()
