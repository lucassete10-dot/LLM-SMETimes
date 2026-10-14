import unittest
from unittest.mock import patch
import numpy as np
from scripts.train_llm_soh_mvp import evaluate,evaluation_curve_metrics
from scripts.train_llm_curve_repair import summary,tiny_gate
from battery_smetimes.llm_curve_objective import curve_metrics

class SparseTrainingMetrics(unittest.TestCase):
    def test_sparse_stage_metrics_keep_numeric_errors_and_fail_joint(self):
        for x in (np.array([101.,200.]),np.array([101.,102.,103.,104.,105.,1000.])):
            y=np.linspace(1.,.8,len(x));p=y+.01
            metric=evaluation_curve_metrics((x,y,p))
            self.assertAlmostEqual(metric['mae_pp'],1.)
            self.assertIsNone(metric['stage_drop_max_pp'])
            self.assertFalse(metric['joint_pass'])
            self.assertFalse(tiny_gate({'sparse':metric},{'tiny_gates_pp':{'stage_drop_max_pp':.5}})['sparse'])

    def test_mixed_train_summary_and_score_do_not_drop_sparse_cell(self):
        arrays={};rows=[]
        for cid,x in [('dense',np.arange(101.,161.)),('sparse',np.array([101.,1000.]))]:
            y=np.linspace(1.,.8,len(x));p=y+.01
            arrays.update({f'{cid}__100__{key}':value for key,value in [('cycle',x),('truth',y),('prediction',p)]})
            rows.append({'cell_id':cid,'origin':100})
        with patch('scripts.train_llm_soh_mvp.evaluate_baseline',return_value=({'by_cell':rows},arrays)):
            _,_,metrics,score=evaluate({}, {}, ['dense','sparse'])
        result=summary(metrics)
        self.assertEqual(result['cells'],2);self.assertEqual(result['metric_cells']['mae_pp'],2)
        self.assertEqual(result['metric_cells']['stage_drop_max_pp'],1)
        self.assertTrue(np.isfinite(score).all())
        self.assertEqual(metrics['dense'],curve_metrics(*(arrays[f'dense__100__{k}'] for k in ('cycle','truth','prediction'))))

if __name__=='__main__':unittest.main()
