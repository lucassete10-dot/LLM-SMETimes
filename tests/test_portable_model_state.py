import unittest
import numpy as np
from scripts.verify_model_state_portable import equal_state

class StateTests(unittest.TestCase):
    def test_random_state(self):
        equal_state(np.random.RandomState(3),np.random.RandomState(3))
        with self.assertRaises(AssertionError):equal_state(np.random.RandomState(3),np.random.RandomState(4))
    def test_generator(self):
        equal_state(np.random.default_rng(3),np.random.default_rng(3))
        with self.assertRaises(AssertionError):equal_state(np.random.default_rng(3),np.random.default_rng(4))
    def test_value_change(self):
        with self.assertRaises(AssertionError):equal_state({'a':np.array([1.,2.])},{'a':np.array([1.,3.])})
    def test_stateless_squared_loss(self):
        from sklearn._loss._loss import CyHalfSquaredError
        equal_state(CyHalfSquaredError(),CyHalfSquaredError())

if __name__=='__main__':unittest.main()
