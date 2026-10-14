import unittest
from scripts.diagnose_llm_shape_bound import variation_mae_bound


class ShapeBoundTests(unittest.TestCase):
    def test_decreasing_curve_has_zero_error(self):
        self.assertAlmostEqual(variation_mae_bound([1.,.9,.8],0.)['global_minimum_mae_pp'],0.)

    def test_isolated_downward_spike_requires_recovery_budget(self):
        self.assertAlmostEqual(variation_mae_bound([1.,.8,1.],0.)['global_minimum_mae_pp'],100*.2/3)
        self.assertAlmostEqual(variation_mae_bound([1.,.8,1.],.1)['global_minimum_mae_pp'],100*.1/3)
        self.assertAlmostEqual(variation_mae_bound([1.,.8,1.],.2)['global_minimum_mae_pp'],0.)


if __name__=='__main__':unittest.main()
