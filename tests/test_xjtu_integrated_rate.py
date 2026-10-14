import unittest
import numpy as np
from battery_smetimes.xjtu_integrated_rate import rate_labels,integrated_drop,predict
from battery_smetimes.xjtu_integrated_rate_v2 import rate_labels as sparse_rate_labels

class RateTests(unittest.TestCase):
    def test_actual_sparse_source_cycles(self):
        c=np.array([101,103,104,108,110,114]);y=1-.001*(c-101)
        _,r=rate_labels(c,y,10);np.testing.assert_allclose(r,.1,atol=1e-12)
    def test_no_invented_labels(self):
        with self.assertRaises(ValueError):rate_labels([101,120],[1,.9],10)
        with self.assertRaises(ValueError):sparse_rate_labels([101,120],[1,.9],10)
    def test_sparse_fallback_actual_intervals(self):
        c=np.array([103,109,115,121]);y=1-.001*(c-103)
        centers,r=sparse_rate_labels(c,y,10)
        np.testing.assert_array_equal(centers,c)
        np.testing.assert_allclose(r,.1,atol=1e-12)
    def test_negative_training_rates_prior(self):
        _,r=rate_labels(np.arange(101,111),np.linspace(.9,1,10),10)
        np.testing.assert_array_equal(r,0)
    def test_integral_and_query_consistency(self):
        fn=lambda c:np.full(len(c),.1)
        c=[101,101.5,107,113.2]
        np.testing.assert_allclose(integrated_drop(c,fn),[0,.0005,.006,.0122])
        np.testing.assert_allclose(integrated_drop(c[:3],fn),integrated_drop(c,fn)[:3])
    def test_acceleration(self):
        p=integrated_drop([101,201,301],lambda c:.001*(c-101))
        self.assertGreater(p[2]-p[1],p[1]-p[0])
    def test_invalid_queries(self):
        for c in ([],[100],[np.inf]):
            with self.assertRaises(ValueError):integrated_drop(c,lambda c:c)
    def test_family_gate(self):
        with self.assertRaises(ValueError):predict(dict(family='A'),dict(physical_family='B'),[101])

if __name__=='__main__':unittest.main()
