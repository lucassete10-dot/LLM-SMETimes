import unittest

import numpy as np

from battery_v2_features import load_cell, load_manifest, make_inference_features
from predict_battery_v2 import predict_series


class FeatureContractTests(unittest.TestCase):
    def test_history_is_causal(self):
        raw = np.arange(40, dtype='float32').reshape(20, 2)
        first = make_inference_features(raw, np.arange(1, 21), 5)
        altered = raw.copy()
        altered[12:] += 1000
        second = make_inference_features(altered, np.arange(1, 21), 5)
        np.testing.assert_array_equal(first.matrix('signal_history')[:12],
                                      second.matrix('signal_history')[:12])

    def test_bad_cycle_or_reference_rejected(self):
        with self.assertRaises(ValueError):
            make_inference_features(np.ones((3, 2)), [1, 1, 2], 1)
        with self.assertRaises(ValueError):
            make_inference_features(np.ones((3, 2)), [1, 2, 3], 4)

    def test_saved_models_predict_without_labels(self):
        seen = set()
        for record in load_manifest()['cells']:
            domain = record['dataset']
            if domain in seen:
                continue
            seen.add(domain)
            cell = load_cell(record)
            result = predict_series(domain, cell.raw, cell.cycle)
            self.assertEqual(len(result['soh']), len(cell.cycle))
            self.assertTrue(np.isnan(result['soh'][:cell.reference_count]).all())
            self.assertTrue(np.isnan(result['rul_cycles'][:15]).all())
            self.assertTrue(np.isfinite(result['soh'][cell.reference_count:]).all())
            self.assertTrue(np.isfinite(result['rul_cycles'][15:]).all())
            self.assertIn('NOT CALIBRATED', result['interval_status'])
            self.assertIsNone(result['interval_coverage_guarantee'])
            self.assertEqual(result['development_band_design_quantile'], .9)
            self.assertIn('soh', result['development_worst_split_coverage'])
            self.assertIn('rul', result['development_worst_split_coverage'])
            self.assertTrue(np.all(result['soh_development_band_lower'][cell.reference_count:]
                                   <= result['soh'][cell.reference_count:]))
            self.assertTrue(np.all(result['rul_development_band_upper_cycles'][15:]
                                   >= result['rul_cycles'][15:]))
            self.assertEqual(len(result['future_cycle_experimental']), 4)
            self.assertEqual(len(result['future_soh_experimental']), 4)
            self.assertEqual(result['future_cycle_experimental'][0], cell.cycle[-1])

    def test_future_strategy_change_suppresses_trajectory_not_point_diagnostic(self):
        record = next(row for row in load_manifest()['cells'] if row['dataset'] == 'NASA')
        cell = load_cell(record)
        same = predict_series('NASA', cell.raw, cell.cycle,
                              future_strategy_relation='same')
        changed = predict_series('NASA', cell.raw, cell.cycle,
                                 future_strategy_relation='changed')
        np.testing.assert_array_equal(same['rul_cycles'], changed['rul_cycles'])
        self.assertIsNone(changed['future_cycle_experimental'])
        self.assertIsNone(changed['future_soh_experimental'])
        self.assertIn('NOT APPLICABLE', changed['rul_future_strategy_status'])
        self.assertIn('CONDITIONAL', same['rul_future_strategy_status'])
        with self.assertRaises(ValueError):
            predict_series('NASA', cell.raw, cell.cycle,
                           future_strategy_relation='invented')


if __name__ == '__main__':
    unittest.main()
