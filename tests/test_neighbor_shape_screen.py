import unittest
from scripts.screen_neighbor_shape import expansion


class ShapeScreenTest(unittest.TestCase):
    def records(self,wins,endpoint=1,mape=1):
        return [dict(selected=dict(score=1 if i<wins else 2,mape_percent=mape,endpoint_pp=endpoint,drop_error_pp=1),raw_control=dict(score=2,mape_percent=1,endpoint_pp=2,drop_error_pp=2)) for i in range(5)]

    def test_requires_four_strict_wins(self):
        self.assertFalse(expansion(self.records(3))['expand'])
        self.assertTrue(expansion(self.records(4))['expand'])

    def test_endpoint_regression_blocks(self):
        self.assertFalse(expansion(self.records(5,endpoint=3))['expand'])

    def test_mape_regression_guard(self):
        self.assertFalse(expansion(self.records(5,mape=1.11))['expand'])


if __name__=='__main__':unittest.main()
