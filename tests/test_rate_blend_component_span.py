import unittest
import numpy as np
from scripts.audit_rate_blend_component_span import span


class ComponentSpanTests(unittest.TestCase):
    def test_same_side_limit(self):
        x=np.array([101.,150.,200.]);y=np.array([1.,.9,.8]);p=np.array([1.,.95,.9]);q=np.array([1.,.94,.88])
        r=span(x,y,p,q);self.assertAlmostEqual(r['outside_fraction'],2/3);self.assertEqual(r['endpoint_same_side'],'both_high');self.assertAlmostEqual(r['endpoint_distance_to_span_pp'],8.)

    def test_bracketed_and_low(self):
        x=np.array([101.,150.,200.]);y=np.array([1.,.9,.8]);p=np.array([1.,.85,.75]);q=np.array([1.,.95,.85])
        r=span(x,y,p,q);self.assertEqual(r['outside_fraction'],0);self.assertEqual(r['endpoint_same_side'],'bracketed')
        r=span(x,y,p,p-.01);self.assertEqual(r['endpoint_same_side'],'both_low');self.assertAlmostEqual(r['endpoint_distance_to_span_pp'],5.)


if __name__=='__main__':unittest.main()
