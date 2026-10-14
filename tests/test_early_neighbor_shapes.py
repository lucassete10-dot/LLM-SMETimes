import unittest
from scripts.audit_early_neighbor_shapes import shape_distance

class ShapeDistanceTest(unittest.TestCase):
    def test_identical(self):
        r=shape_distance([101,201,301],[1,.98,.8],[101,201,301],[1,.98,.8])
        for f in ('mean_abs_soh_difference_pp','drop_difference_pp','acceleration_difference_pp'):self.assertAlmostEqual(r[f],0)
    def test_offset_not_drop_or_acceleration(self):
        r=shape_distance([101,201,301],[1,.98,.8],[101,201,301],[1.02,1.,.82])
        self.assertAlmostEqual(r['mean_abs_soh_difference_pp'],2);self.assertAlmostEqual(r['drop_difference_pp'],0);self.assertAlmostEqual(r['acceleration_difference_pp'],0)
    def test_no_extrapolation(self):
        r=shape_distance([101,301],[1,.8],[201,401],[.9,.7]);self.assertEqual(r['common_start'],201);self.assertEqual(r['common_end'],301)
        self.assertIsNone(shape_distance([101,201],[1,.9],[301,401],[.8,.7]))

if __name__=='__main__':unittest.main()
