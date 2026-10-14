import unittest
import numpy as np
from battery_smetimes.llm_v3_reporting import enrich_quality_rows,quality_summary
from battery_smetimes.llm_v3_workflow import SEEDS


class ReportingV3Tests(unittest.TestCase):
    def row(self,seed,mape=1.,stage=1.):
        return {'cell_id':'a','fold':'f','seed':seed,'mape_percent':mape,'late_mape_percent':2.,
            'start_error_pp':.5,'endpoint_error_pp':1.,'stage_drop_max_pp':stage}

    def test_stability_requires_third_primary_pass_and_two_quality_passes(self):
        rows=[self.row(SEEDS[0]),self.row(SEEDS[1]),self.row(SEEDS[2],2.)]
        r=quality_summary(rows,3)
        self.assertEqual(r['reference_numeric_passes'],2)
        self.assertEqual(r['stable_reference_cell_fold_groups'],1)
        rows[2]['mape_percent']=4.
        self.assertEqual(quality_summary(rows,3)['stable_reference_cell_fold_groups'],0)

    def test_missing_seed_and_sparse_stage_remain_in_denominator(self):
        rows=[self.row(SEEDS[0],stage=None),self.row(SEEDS[1])]
        r=quality_summary(rows,3)
        self.assertEqual(r['missing_cell_seed_instances'],1)
        self.assertEqual(r['reference_pass_fraction'],1/3)
        self.assertEqual(r['stage_evidence_unavailable_instances'],1)
        self.assertEqual(r['stable_reference_cell_fold_groups'],0)

    def test_raw_late_mape_replaces_cached_optimistic_values(self):
        rows=[self.row(SEEDS[0])];x=np.arange(101,141);y=np.ones(40);p=y.copy();p[-8:]=.9
        archive={'a':{'test':[{'fold':'f','seed':SEEDS[0],'x':x,'y':y,'p':p}]}}
        enrich_quality_rows(rows,archive,'test')
        self.assertAlmostEqual(rows[0]['late_mape_percent'],10.)
        self.assertFalse(rows[0]['reference_numeric_pass'])
        self.assertIn('late_mape_percent',rows[0]['reference_quality_failures'])

    def test_duplicate_seed_is_rejected(self):
        with self.assertRaises(ValueError):quality_summary([self.row(SEEDS[0])]*2,2)


if __name__=='__main__':unittest.main()
