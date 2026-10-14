import unittest
import numpy as np
from battery_smetimes.llm_curve_quality_v3 import (
    freeze_validation_policy, validation_score, curve_metrics_v3, reference_pass)


class QualityV3Tests(unittest.TestCase):
    def cell(self, x):
        return {'label_cycle': np.asarray(x), 'soh': np.ones(len(x))}

    def test_more_strict_passes_cannot_override_curve_error(self):
        policy = freeze_validation_policy({'a': self.cell(np.arange(101,141))}, ['a'], 100)
        low = curve_metrics_v3(np.arange(101,141), np.ones(40), np.ones(40)*.99)
        high = curve_metrics_v3(np.arange(101,141), np.ones(40), np.ones(40)*.9)
        low['joint_pass'] = False
        high['joint_pass'] = True
        self.assertLess(validation_score({'a': low}, policy), validation_score({'a': high}, policy))

    def test_sparse_stage_is_removed_for_whole_frozen_split(self):
        cells = {'a': self.cell(np.arange(101,141)), 'b': self.cell([101,140])}
        policy = freeze_validation_policy(cells, ['a','b'], 100)
        self.assertNotIn('stage_drop_max_pp', policy['active_metrics'])
        self.assertEqual(policy['stage_unavailable_cells'], ['b'])
        metrics = {cid: curve_metrics_v3(c['label_cycle'], c['soh'], c['soh']) for cid,c in cells.items()}
        self.assertEqual(validation_score(metrics, policy), 0.)
        self.assertFalse(reference_pass(metrics['b']))

    def test_missing_prediction_and_nonfinite_metric_fail(self):
        policy = freeze_validation_policy({'a': self.cell([101,140])}, ['a'], 100)
        with self.assertRaises(ValueError): validation_score({}, policy)
        row = curve_metrics_v3([101,140], [1.,1.], [1.,1.])
        row['tail_mae_pp'] = float('nan')
        with self.assertRaises(ValueError): validation_score({'a':row}, policy)

    def test_late_mape_and_pp_use_distinct_units(self):
        row = curve_metrics_v3([101,120,140], [1.,.8,.5], [1.,.8,.49])
        self.assertAlmostEqual(row['tail_mae_pp'], 1.)
        self.assertAlmostEqual(row['late_mape_percent'], 2.)

    def test_worst_fraction_is_ceil_not_single_worst_only(self):
        cells = {str(i):self.cell([101,140]) for i in range(6)}
        policy = freeze_validation_policy(cells, list(cells), 100)
        metrics = {cid:{key:value*scale for key,scale in policy['scales'].items()}
                   for cid,value in zip(cells,[0.,0.,0.,0.,2.,4.])}
        self.assertAlmostEqual(validation_score(metrics,policy),1.75)


if __name__ == '__main__': unittest.main()
