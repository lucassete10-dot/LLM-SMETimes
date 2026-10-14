import unittest
from scripts.verify_neighbor_curve_nca_fivefold import decision

class ExpansionTest(unittest.TestCase):
    def records(self,wins,endpoint=1):
        control=dict(score=2,endpoint_pp=2,drop_error_pp=2,mape_percent=1)
        return [dict(selected=dict(score=1 if i<wins else 2,endpoint_pp=endpoint,drop_error_pp=1,mape_percent=1),control=control) for i in range(5)]
    def test_requires_four_wins(self):
        self.assertFalse(decision(self.records(3))[2]);self.assertTrue(decision(self.records(4))[2])
    def test_error_regression_blocks(self):
        self.assertFalse(decision(self.records(5,3))[2])

if __name__=='__main__':unittest.main()
