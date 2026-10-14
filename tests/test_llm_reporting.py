import unittest
from battery_smetimes.llm_reporting import summarize_curve_rows,raw_curve_numeric_metrics


class ReportingTests(unittest.TestCase):
    def row(self,cid,mape,mae,shape=True):
        return {'cell_id':cid,'fold':'fold1','seed':1,'mape_percent':mape,'mae_pp':mae,'evaluable':shape,'strict_joint_pass':False}

    def test_all_requested_denominator_keeps_missing_and_sparse_shape_unknown(self):
        rows=[self.row('a',4.,1.,False),self.row('b',6.,1.),self.row('c',3.,3.)]
        result=summarize_curve_rows(rows,4,{'mape_percent':5.,'mae_pp':2.})
        self.assertEqual(result['mape_target_passes'],2);self.assertEqual(result['practical_numeric_passes'],1)
        self.assertEqual(result['mape_target_pass_fraction'],.5);self.assertEqual(result['missing_cell_seed_instances'],1)
        self.assertEqual(result['metric_distributions']['mape_percent']['median'],4.)
        self.assertEqual(result['metric_distributions']['mape_percent']['worst_instances'][0]['cell_id'],'b')

    def test_empty_or_missing_metrics_never_produce_nan_or_false_pass(self):
        result=summarize_curve_rows([],3,{'mape_percent':5.,'mae_pp':2.})
        self.assertIsNone(result['metric_distributions']['mape_percent']['mean'])
        row=self.row('a',float('nan'),None)
        result=summarize_curve_rows([row],1,{'mape_percent':5.,'mae_pp':2.})
        self.assertEqual(result['mape_target_passes'],0);self.assertEqual(result['practical_numeric_passes'],0)
        self.assertIsNone(result['mape_failures'][0]['mape_percent'])

    def test_extra_predictions_cannot_silently_change_denominator(self):
        with self.assertRaises(ValueError):summarize_curve_rows([self.row('a',1,1)],0,{'mape_percent':5.})

    def test_sparse_numeric_accuracy_does_not_invent_stage_measurements(self):
        values=raw_curve_numeric_metrics([101,200],[1.,.8],[.99,.82])
        self.assertAlmostEqual(values['mape_percent'],1.75)
        self.assertAlmostEqual(values['mae_pp'],1.5)
        self.assertIsNone(values['stage_drop_max_pp']);self.assertFalse(values['stage_metrics_available'])
        row={'cell_id':'sparse','fold':'fold1','seed':1,**values}
        summary=summarize_curve_rows([row],1,{'mape_percent':5.,'stage_drop_max_pp':3.})
        self.assertEqual(summary['mape_target_passes'],1);self.assertEqual(summary['practical_numeric_passes'],0)

    def test_dense_numeric_metrics_match_existing_arithmetic(self):
        import numpy as np
        from battery_smetimes.llm_curve_objective import curve_metrics
        x=np.arange(101,401);y=1.-(x-101)*.0004;p=y+.005*np.sin(x/15)
        old=curve_metrics(x,y,p);numeric=raw_curve_numeric_metrics(x,y,p)
        for key in ('mape_percent','mae_pp','tail_mae_pp','start_error_pp','endpoint_error_pp','stage_drop_max_pp'):
            self.assertAlmostEqual(old[key],numeric[key],places=10)


if __name__=='__main__':unittest.main()
