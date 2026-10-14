import unittest
import numpy as np
from scripts.audit_curve_error_components import decompose,summarize

class ComponentsTest(unittest.TestCase):
    def test_offset_only(self):
        r=decompose([101,200,300],[1,.95,.9],[1.02,.97,.92])
        self.assertAlmostEqual(r['start_abs_pp'],2);self.assertAlmostEqual(r['change_abs_pp'],0);self.assertAlmostEqual(r['mse_pp2'],4)
    def test_shape_and_cancellation(self):
        r=decompose([101,200,300],[1,.95,.9],[1.02,.96,.90])
        self.assertTrue(r['endpoint_cancellation']);self.assertAlmostEqual(r['change_abs_pp'],2)
        self.assertAlmostEqual(r['mse_pp2'],r['offset_square_pp2']+r['shape_square_pp2']+r['cross_term_pp2'])
    def test_equal_cell_not_seed_weighting(self):
        r=decompose([101,200],[1,.9],[1.02,.92]);s=decompose([101,200],[1,.9],[1.04,.94])
        summary,_=summarize([dict(fold='f',cell_id='a',**r)]*5+[dict(fold='f',cell_id='b',**s)])
        self.assertAlmostEqual(summary['start_abs_pp'],3);self.assertEqual(summary['cell_count'],2)
    def test_bad_input(self):
        for c,y,p in [([102,101],[1,.9],[1,.9]),([101],[1],[1]),([101,102],[1,np.nan],[1,.9])]:
            with self.assertRaises(ValueError):decompose(c,y,p)
    def test_underacceleration(self):
        r=decompose([101,201,301],[1,.98,.8],[1,.9,.8])
        self.assertAlmostEqual(r['true_secondhalf_extra_drop_pp'],16)
        self.assertAlmostEqual(r['pred_secondhalf_extra_drop_pp'],0)
        self.assertAlmostEqual(r['acceleration_abs_error_pp'],16);self.assertTrue(r['underaccelerates'])

if __name__=='__main__':unittest.main()
