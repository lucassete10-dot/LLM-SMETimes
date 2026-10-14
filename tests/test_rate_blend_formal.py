"""Formal evaluation saves entire curves; no query label enters components."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.train_rate_blend_formal import components,evaluate
from scripts.verify_rate_blend import score


class FormalRateBlendTests(unittest.TestCase):
    def test_complete_curve_and_label_exclusion(self):
        cell=dict(physical_family='MIT',cycle=np.arange(1.,101.),raw=np.ones((100,51)),soh=np.array([.99,.95,.8]),label_cycle=np.array([101.,120.,180.]))
        parts=dict(primary=dict(family='MIT'),rate=dict(family='MIT'))
        def primary(b,c,cycle):
            if c['physical_family']!=b['family']:raise ValueError('family')
            if 'soh' in c:self.assertTrue(np.isinf(c['soh']))
            if 'label_cycle' in c:raise AssertionError('query labels leaked')
            take=np.asarray(c['cycle'])<=100;self.assertTrue(np.isfinite(c['raw'][take]).all())
            return np.array([1.,.96,.88])
        def rate(b,c,cycle):return primary(b,c,cycle)-.02
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.train_rate_blend_formal.primary_predict',side_effect=primary),patch('scripts.train_rate_blend_formal.rate_predict',side_effect=rate),patch('scripts.train_rate_blend_formal.labels',return_value=(cell['label_cycle'],cell['soh'])):
            folder=Path(tmp);a=components(parts,{'cell':cell},['cell'],folder,'validation');rows=evaluate(a,['cell'],.5,folder,'validation')
            with np.load(folder/'validation_predictions.npz') as arrays:
                self.assertEqual(len(arrays.files),3);np.testing.assert_array_equal(arrays['cell__100__cycle'],cell['label_cycle'])
                np.testing.assert_allclose(arrays['cell__100__prediction'],[.99,.95,.87],atol=1e-12,rtol=0)
            self.assertEqual(json.loads((folder/'validation_metrics.json').read_text())[0]['cell_id'],'cell')
            self.assertAlmostEqual(rows[0]['endpoint_pp'],7.)

    def test_fixed_score_and_tie_order(self):
        rows=[dict(mape_percent=1.,late_mape=2.,endpoint_pp=3.,drop_error_pp=4.)]
        self.assertAlmostEqual(score(rows),3.125)
        candidates=[dict(score=2.,weight=.5),dict(score=2.,weight=0.)]
        self.assertEqual(min(candidates,key=lambda r:(r['score'],r['weight']))['weight'],0.)


if __name__=='__main__':unittest.main()
