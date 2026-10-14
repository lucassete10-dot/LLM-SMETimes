import unittest
import numpy as np
from scripts.audit_tree_leaf_support import donors


class LeafSupportTests(unittest.TestCase):
    def test_weighted_multi_tree_membership(self):
        leaves=np.array([[1,3],[1,4],[2,3]])
        actual=donors(leaves,np.array([1,3]),np.array([1.,3.,2.]))
        expected=np.array([.5*(1/4+1/3),.5*3/4,.5*2/3])
        np.testing.assert_allclose(actual,expected,atol=1e-15,rtol=0)

    def test_missing_leaf_rejected(self):
        with self.assertRaises(AssertionError):
            donors(np.array([[1],[2]]),np.array([9]),np.ones(2))


if __name__=='__main__':unittest.main()
