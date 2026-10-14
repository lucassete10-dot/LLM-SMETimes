import unittest
import numpy as np
from battery_smetimes.neighbor_rate_shape import pace,map_cycle


class RateShapeTest(unittest.TestCase):
    def test_speed_ratio_and_clamp(self):
        self.assertAlmostEqual(pace(1,.9,1,.95),2)
        self.assertAlmostEqual(pace(1,.95,1,.9),.5)
        self.assertEqual(pace(1,.7,1,.99),2)

    def test_flat_or_reverse_fallback(self):
        self.assertEqual(pace(1,1,1,.9),1)
        self.assertEqual(pace(1,.9,1,1.1),1)

    def test_mapping_preserves_anchor_and_changes_shape(self):
        np.testing.assert_array_equal(map_cycle([101,151,201],2),[101,201,301])

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):map_cycle([100],1)
        with self.assertRaises(ValueError):pace(1,np.nan,1,.9)


if __name__=='__main__':unittest.main()
