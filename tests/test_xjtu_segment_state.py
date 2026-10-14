import unittest
import numpy as np
from scripts.verify_xjtu_segment_fold_state import equal_state

class ExactStateTests(unittest.TestCase):
    def test_dtype_and_arrays(self):
        equal_state({'dtype':np.dtype('float64'),'x':np.array([1.,np.nan])},{'dtype':np.dtype('float64'),'x':np.array([1.,np.nan])})
    def test_different_values_are_not_accepted(self):
        with self.assertRaises(AssertionError):equal_state(np.array([1.]),np.array([1.+1e-12]))
    def test_different_metadata_is_not_accepted(self):
        with self.assertRaises(AssertionError):equal_state({'ids':['a','b']},{'ids':['a','c']})

if __name__=='__main__':unittest.main()
