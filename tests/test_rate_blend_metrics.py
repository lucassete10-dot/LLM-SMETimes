import unittest
import numpy as np
from scripts.audit_rate_blend_metrics import metrics,summarize,checks,FIELDS


class RateBlendMetricsTests(unittest.TestCase):
    def test_exact_metrics(self):
        m=metrics(np.array([101.,120.,150.]),np.array([1.,.9,.8]),np.array([1.,.88,.85]))
        self.assertAlmostEqual(m['endpoint_pp'],5.);self.assertAlmostEqual(m['drop_error_pp'],5.);self.assertAlmostEqual(m['late_mape'],6.25)
        with self.assertRaises(AssertionError):metrics(np.array([101.,101.]),np.ones(2),np.ones(2))

    def test_balanced_cells_and_seed_sd(self):
        rows=[dict(cell_id=c,seed=s,**{f:float(s) for f in FIELDS}) for c in ('a','b') for s in range(5)]
        summary,cells=summarize(rows);self.assertEqual(summary['cells'],2);self.assertAlmostEqual(summary['endpoint_pp']['mean'],2.)
        self.assertAlmostEqual(summary['endpoint_pp']['seed_sd'],np.std(np.arange(5),ddof=1));self.assertTrue(all(c['joint_pass'] for c in cells))
        with self.assertRaises(AssertionError):summarize(rows+[rows[0]])
        old=dict(summary,endpoint_p90=2.,drop_error_p90=2.);new=dict(summary,endpoint_p90=3.,drop_error_p90=3.)
        self.assertTrue(checks(new,old)['endpoint_p90']);new['endpoint_p90']=3.01;self.assertFalse(checks(new,old)['endpoint_p90'])


if __name__=='__main__':unittest.main()
