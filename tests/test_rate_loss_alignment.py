import unittest
from unittest.mock import patch
import numpy as np
from scripts.train_rate_loss_alignment import aligned_weights

class LossAlignmentTest(unittest.TestCase):
    def test_cell_mass_and_loss_identity(self):
        cells={'a':np.array([101.,200.,400.]),'b':np.array([101.,900.])}
        with patch('scripts.train_rate_loss_alignment.labels',side_effect=lambda c:(c,np.ones(len(c)))):
            w=aligned_weights(cells,['a','b'],2)
        self.assertAlmostEqual(w[:3].sum(),w[3:].sum());self.assertAlmostEqual(w.mean(),1)
        for c,v in [(cells['a'],w[:3]),(cells['b'],w[3:])]:
            base=np.linspace(1,3,len(c));d=c/100;residual=np.linspace(.01,.03,len(c));constant=v[0]/(base[0]*d[0]**2)
            self.assertAlmostEqual(float(np.sum(v*residual**2)),float(constant*np.sum(base*(d*residual)**2)))
    def test_power_rejected(self):
        with self.assertRaises(ValueError):aligned_weights({},[],3)

if __name__=='__main__':unittest.main()
