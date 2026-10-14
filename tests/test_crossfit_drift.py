import unittest
import numpy as np
from battery_smetimes.crossfit_drift import split_ids,basis,fit_head


class DriftTest(unittest.TestCase):
    def test_whole_cell_oof(self):
        ids=['a','b','c','d','e'];splits=split_ids(ids,4)
        self.assertEqual(sorted(k for s in splits for k in s['held']),sorted(ids))
        for s in splits:
            self.assertFalse(set(s['train'])&set(s['held']))
            self.assertEqual(set(s['train']+s['held']),set(ids))
        self.assertEqual(splits,split_ids(ids,4))

    def test_basis_is_source_cycle_not_life_ratio(self):
        np.testing.assert_allclose(basis([200,300],2),[[1,1],[2,4]])
        with self.assertRaises(ValueError):basis([100],1)
        with self.assertRaises(ValueError):split_ids(['a'],0)

    def test_head_training_scaler(self):
        head=fit_head(np.array([101,200,300]),np.array([0,.01,.02]),np.ones(3),1,10)
        np.testing.assert_allclose(head['scaler'].mean_,[np.mean([.01,1,2])])


if __name__=='__main__':unittest.main()
