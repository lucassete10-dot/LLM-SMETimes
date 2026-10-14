import unittest
import numpy as np
from scripts.audit_neighbor_selection_stability import scores

class ScoreSensitivityTest(unittest.TestCase):
    def test_score_cell_resampling(self):
        values=np.array([[1.,2.,3.],[3.,2.,1.]])
        np.testing.assert_allclose(scores(values,[0,1,2]),[2.7,2.7])
        np.testing.assert_allclose(scores(values,[0,0,0]),[1.25,3.75])

if __name__=='__main__':unittest.main()
