import unittest
from scripts.optimize_llm_dataset_validation import validation_summary,passes_screen,confirmed


class ValidationSelectionTests(unittest.TestCase):
    def test_configured_three_percent_target_controls_validation_pass_count(self):
        plan=self.plan();plan['numeric_thresholds']['mape_percent']=3.
        metric={'mae_pp':1.,'tail_mae_pp':2.,'stage_drop_max_pp':2.,'mape_percent':3.1}
        self.assertEqual(validation_summary({'a':metric},plan)['mape_passes'],0)
        metric['mape_percent']=3.
        self.assertEqual(validation_summary({'a':metric},plan)['mape_passes'],1)
    def test_sparse_missing_stage_cannot_be_ranked_or_confirmed(self):
        metrics={'sparse':{'mae_pp':1.,'tail_mae_pp':2.,'stage_drop_max_pp':None,'stage_metrics_available':False,'mape_percent':4.}}
        value=validation_summary(metrics,self.plan())
        self.assertIsNone(value['score']);self.assertFalse(value['ranking_eligible'])
        self.assertEqual(value['mape_passes'],1);self.assertEqual(value['stage_unavailable_cells'],['sparse'])
        self.assertFalse(passes_screen(value,self.row(),self.plan()))
        self.assertFalse(confirmed([value]*3,[self.row()]*3,self.plan()))
    def plan(self):
        return {'score_weights':{'mae_pp':1.,'tail_mae_pp':1.,'stage_drop_max_pp':.5},
            'numeric_thresholds':{'mape_percent':5.,'mae_pp':2.},'screen_min_relative_improvement':.1,
            'confirmation_min_mean_relative_improvement':.1,'confirmation_min_improved_seed_count':2,'worst_val_mae_max_ratio':1.1}

    def row(self,score=10.,worst=4.,passes=3,cells=4):
        return {'score':score,'worst_mae_pp':worst,'mape_passes':passes,'cells':cells}

    def test_mean_improvement_cannot_hide_worst_cell_regression_or_denominator_change(self):
        base=self.row();self.assertTrue(passes_screen(self.row(score=8),base,self.plan()))
        self.assertFalse(passes_screen(self.row(score=8,worst=5),base,self.plan()))
        self.assertFalse(passes_screen(self.row(score=8,passes=2),base,self.plan()))
        self.assertFalse(passes_screen(self.row(score=8,cells=3),base,self.plan()))

    def test_single_lucky_seed_cannot_confirm_algorithm(self):
        base=[self.row() for _ in range(3)]
        self.assertFalse(confirmed([self.row(score=4),self.row(score=11),self.row(score=11)],base,self.plan()))
        self.assertTrue(confirmed([self.row(score=8),self.row(score=8),self.row(score=10)],base,self.plan()))

    def test_mape_target_is_five_and_missing_validation_cell_is_rejected(self):
        metrics={'a':{'evaluable':True,'mae_pp':1.,'tail_mae_pp':2.,'stage_drop_max_pp':2.,'mape_percent':4.9},
            'b':{'evaluable':True,'mae_pp':1.,'tail_mae_pp':2.,'stage_drop_max_pp':2.,'mape_percent':5.1}}
        value=validation_summary(metrics,self.plan());self.assertEqual(value['mape_passes'],1);self.assertEqual(value['score'],4.)
        metrics['b']['evaluable']=False
        self.assertEqual(validation_summary(metrics,self.plan())['cells'],2)
        del metrics['b']['mae_pp']
        with self.assertRaisesRegex(ValueError,'coverage'):validation_summary(metrics,self.plan())


if __name__=='__main__':unittest.main()
