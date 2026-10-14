import unittest
from unittest.mock import patch
import numpy as np
from scripts.train_nasa_robust_inner import specifications,predict,detail,decisions


class RobustInnerTests(unittest.TestCase):
    def test_fixed_budget_and_parameter_mapping(self):
        s=specifications();self.assertEqual(len(s),25);self.assertEqual(s[17]['calibration'],dict(mode='absolute',alpha=100.,window=30));self.assertEqual(s[23]['calibration'],dict(mode='delta',alpha=100.,window=30));self.assertEqual(s[1]['calibration'],s[13]['calibration'])

    def test_level_preserves_drop(self):
        def base(bundle,query,origin,c):return 1-.001*np.asarray(c)
        with patch('scripts.train_nasa_robust_inner.predict_baseline',side_effect=base),patch('scripts.train_nasa_robust_inner.calpredict',return_value=np.array([.95])):
            p=predict({}, {}, 'level',{},[101,151,201]);np.testing.assert_allclose(p,[.95,.90,.85],rtol=0,atol=1e-12)

    def test_short_coverage_and_constant_truth_are_json_safe(self):
        import json
        r=detail(np.array([101.,102.]),np.array([1.,1.]),np.array([1.,1.]));self.assertIsNone(r['r2']);self.assertTrue(r['short_coverage']);json.dumps(r,allow_nan=False)

    def test_selector_uses_all_heldout_cells(self):
        rr=[]
        for i in range(25):
            for j,key in enumerate(['a','b','c']):
                value=([0,20,20][j] if i==1 else 3 if i==2 else 10)
                rr.append(dict(cell_id=key,candidate=i,**{f:value for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
        d=decisions(rr,1);self.assertEqual(d['selected']['candidate'],2);self.assertEqual(d['single_validation_choice']['candidate'],1)


if __name__=='__main__':unittest.main()
