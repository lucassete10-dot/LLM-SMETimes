import unittest
from scripts.summarize_public_soh_validation import aggregate

class SummaryTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for seed in range(5):
            for name,error in [('short',2.),('long',10.)]:
                full=dict(mape_percent=error,mae_soh=.01,rmse_soh=.02,r2=.5)
                tail=dict(full,r2=None)
                rows.append(dict(cell_id=name,seed=seed,metrics=full,late_half=full,last100=tail,
                    endpoint_abs_soh_pp=1.,future_targets=2 if name=='short' else 1000))
        return rows

    def test_equal_cell_macro_not_observation_pooling(self):
        s=aggregate(self.rows())
        self.assertEqual(s['metrics']['mape_percent']['mean'],6.)
        self.assertEqual(s['fraction_cells_mean_mape_le6'],.5)
        self.assertEqual([r['cell_id'] for r in s['failures_by_mean_mape']],['long'])

    def test_undefined_r2_stays_undefined(self):
        s=aggregate(self.rows())
        r=s['last100']['r2']
        self.assertIsNone(r['mean']);self.assertIsNone(r['seed_sd'])
        self.assertEqual(r['defined_cell_seed_count'],0)
        self.assertEqual(r['seed_values'],[None]*5)

if __name__=='__main__':unittest.main()
