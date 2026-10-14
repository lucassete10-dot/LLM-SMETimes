import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import external_test_v2
from external_test_v2 import (FEATURE_SCHEMAS, predict, read_signal, score,
                              score_cell, score_rul_challenger, paired_soh_summary,
                              paired_rul_summary, signal_fingerprint,
                              reject_training_overlap, validate_diagnostic_rate,
                              validate_signal_manifest)


class ExternalTestWorkflowTests(unittest.TestCase):
    def test_raw_rul_challenger_respects_endpoint_censoring(self):
        cycle = np.arange(1, 21)
        estimate = np.full(len(cycle), np.nan)
        estimate[15:] = 20-cycle[15:]
        prediction = {'cycle': cycle, 'rul_challenger_raw_target': estimate}
        self.assertIsNone(score_rul_challenger(prediction, None))
        scored = score_rul_challenger(prediction, 20)
        self.assertEqual(scored['observations'], 5)
        self.assertEqual(scored['mae_cycles'], 0.)
        self.assertEqual(scored['interval_status'], 'NO CALIBRATED INTERVAL')

    def test_raw_rul_pairing_uses_cells_not_cycles(self):
        phase = {'early': 10., 'middle': 5., 'late': 2.}
        rows = [
            {'cell_id': 'new_a', 'aging_strategy_relation': 'known',
             'future_strategy_relation': 'same',
             'rul': {'mae_cycles': 20., 'signed_bias_cycles': 10.,
                     'phase_mae_cycles': phase},
             'rul_challenger_raw_target': {'mae_cycles': 10.,
                 'signed_bias_cycles': 5., 'phase_mae_cycles': phase}},
            {'cell_id': 'new_b', 'aging_strategy_relation': 'novel',
             'future_strategy_relation': 'changed',
             'rul': {'mae_cycles': 10., 'signed_bias_cycles': -2.,
                     'phase_mae_cycles': phase},
             'rul_challenger_raw_target': {'mae_cycles': 15.,
                 'signed_bias_cycles': 4., 'phase_mae_cycles': phase}},
        ]
        report = paired_rul_summary(rows)
        self.assertEqual(report['cells'], 2)
        self.assertEqual(report['mae_delta_challenger_minus_baseline_cycles'], -2.5)
        self.assertEqual((report['cells_improved'], report['cells_worsened']), (1, 1))
        self.assertEqual(report['cell_bootstrap_status'], 'INSUFFICIENT_CELLS')
        self.assertIn('changed', report['by_future_strategy_relation'])

    def test_diagnostic_rate_must_match_claimed_protocol(self):
        with self.assertRaisesRegex(ValueError, 'diagnostic discharge rate'):
            validate_diagnostic_rate('new_mit', 'MIT',
                                     {'diagnostic_discharge_rate_c': None})
        with self.assertRaisesRegex(ValueError, 'diagnostic discharge rate'):
            validate_diagnostic_rate('new_mit', 'MIT',
                                     {'diagnostic_discharge_rate_c': 1.})
        with self.assertRaisesRegex(ValueError, 'diagnostic discharge rate'):
            validate_diagnostic_rate('new_xjtu', 'XJTU',
                                     {'diagnostic_discharge_rate_c': .2})
        for domain, rate in (('CALCE', 1.), ('MIT', 4.), ('XJTU', .5), ('XJTU', 1.)):
            validate_diagnostic_rate('new_cell', domain,
                                     {'diagnostic_discharge_rate_c': rate})
        validate_diagnostic_rate('new_nasa', 'NASA',
                                 {'diagnostic_discharge_rate_c': None})
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'signals.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'cells': [{
                'cell_id': 'new_mit_wrong_rate', 'dataset': 'MIT',
                'feature_schema': FEATURE_SCHEMAS['MIT'][0],
                'cycle_unit': FEATURE_SCHEMAS['MIT'][2],
                'protocol_description': 'synthetic mismatch fixture',
                'diagnostic_protocol_match': True,
                'aging_strategy_relation': 'novel',
                'operating_conditions': {'charge_rate_c': None,
                    'discharge_rate_c': None, 'diagnostic_discharge_rate_c': 1.,
                    'ambient_temperature_c': None, 'strategy': 'synthetic policy'},
                'signal_file': 'not_needed_for_this_validation.npz'}]}))
            with self.assertRaisesRegex(ValueError, 'diagnostic discharge rate'):
                validate_signal_manifest(manifest)

    def test_paired_soh_summary_counts_cells_and_omits_small_bootstrap(self):
        rows = [
            {'cell_id': 'new_a', 'aging_strategy_relation': 'known',
             'soh': {'mae': .10, 'signed_bias': .05},
             'soh_challenger_300': {'mae': .08, 'signed_bias': .04}},
            {'cell_id': 'new_b', 'aging_strategy_relation': 'novel',
             'soh': {'mae': .02, 'signed_bias': -.01},
             'soh_challenger_300': {'mae': .05, 'signed_bias': .02}},
        ]
        report = paired_soh_summary(rows)
        self.assertEqual(report['cells'], 2)
        self.assertAlmostEqual(report['mae_delta_challenger_minus_baseline'], .005)
        self.assertEqual((report['cells_improved'], report['cells_worsened'],
                          report['cells_tied']), (1, 1, 0))
        self.assertEqual(report['cell_bootstrap_status'], 'INSUFFICIENT_CELLS')
        self.assertIsNone(report['cell_bootstrap_95pct_mae_delta_interval'])
        self.assertAlmostEqual(report['per_cell_mae_delta']['new_a'], -.02)
        self.assertEqual(report['by_aging_strategy_relation']['novel']['cells'], 1)

    def test_paired_soh_bootstrap_resamples_whole_cells_deterministically(self):
        rows = [
            {'cell_id': f'new_{i}', 'aging_strategy_relation': 'novel',
             'soh': {'mae': .10 + i*.001, 'signed_bias': .02},
             'soh_challenger_300': {'mae': .08 + i*.001,
                                    'signed_bias': .01}}
            for i in range(8)]
        first = paired_soh_summary(rows)
        self.assertEqual(first, paired_soh_summary(list(reversed(rows))))
        self.assertEqual(first['cells_improved'], 8)
        self.assertEqual(first['cell_bootstrap_status'], 'DESCRIPTIVE_ONLY')
        lower, upper = first['cell_bootstrap_95pct_mae_delta_interval']
        self.assertAlmostEqual(lower, -.02)
        self.assertAlmostEqual(upper, -.02)

    def test_signal_file_rejects_embedded_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.npz'
            np.savez_compressed(path, raw_features=np.ones((20, 44)),
                                cycle=np.arange(1, 21), capacity_ah=np.ones(20))
            with self.assertRaisesRegex(ValueError, 'label-like'):
                read_signal(path, 'NASA')

    def test_raw_prefix_overlap_survives_cycle_renumbering(self):
        raw = np.arange(20*44, dtype='float32').reshape(20, 44)
        first = signal_fingerprint(np.arange(1, 21), raw, True, True)
        second = signal_fingerprint(np.arange(10, 30), raw, True, True)
        self.assertEqual(first, second)

    def test_xjtu_challenger_training_prefix_is_rejected(self):
        with np.load('battery_models_v2/xjtu_long_prefix/features.npz') as saved:
            cycle = saved['b5_RW_battery-1_cycle']
            raw = saved['b5_RW_battery-1_300s']
        row = {'cell_id': 'new_id_with_training_300s_prefix',
               'feature_fingerprint': 'new-full',
               'prefix_fingerprint': 'new-prefix',
               'raw_prefix_fingerprint': 'new-raw-prefix',
               'challenger_raw_prefix_fingerprint': signal_fingerprint(
                   cycle + 1000, raw, True, True)}
        with self.assertRaisesRegex(ValueError, 'challenger signal rows'):
            reject_training_overlap([row])

    def test_duplicate_signal_cannot_count_as_two_external_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez_compressed(root / 'signal.npz', raw_features=np.ones((20, 44)),
                                cycle=np.arange(1, 21))
            template = {'dataset': 'NASA', 'feature_schema': FEATURE_SCHEMAS['NASA'][0],
                        'cycle_unit': FEATURE_SCHEMAS['NASA'][2],
                        'protocol_description': 'synthetic smoke data',
                        'diagnostic_protocol_match': True,
                        'aging_strategy_relation': 'novel',
                        'operating_conditions': {'charge_rate_c': None,
                            'discharge_rate_c': None, 'diagnostic_discharge_rate_c': None,
                            'ambient_temperature_c': None, 'strategy': 'synthetic'},
                        'signal_file': 'signal.npz'}
            manifest = root / 'signals.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'cells': [
                {**template, 'cell_id': 'external_a'},
                {**template, 'cell_id': 'external_b'}]}))
            with self.assertRaisesRegex(ValueError, 'reuses a signal'):
                validate_signal_manifest(manifest)

    def test_changed_future_strategy_requires_description(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez_compressed(root / 'signal.npz', cycle=np.arange(1, 21),
                                raw_features=np.ones((20, 44), dtype='float32'))
            manifest = root / 'signals.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'cells': [{
                'cell_id': 'new_future_strategy', 'dataset': 'NASA',
                'feature_schema': FEATURE_SCHEMAS['NASA'][0],
                'cycle_unit': FEATURE_SCHEMAS['NASA'][2],
                'protocol_description': 'synthetic smoke only',
                'diagnostic_protocol_match': True, 'aging_strategy_relation': 'novel',
                'operating_conditions': {'charge_rate_c': None, 'discharge_rate_c': None,
                    'diagnostic_discharge_rate_c': None, 'ambient_temperature_c': None,
                    'strategy': 'synthetic observed'},
                'future_strategy_relation': 'changed',
                'signal_file': 'signal.npz'}]}))
            with self.assertRaisesRegex(ValueError, 'requires a description'):
                validate_signal_manifest(manifest)

    def test_failed_blind_prediction_never_publishes_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez_compressed(root / 'signal.npz', cycle=np.arange(1, 21),
                                raw_features=np.ones((20, 44), dtype='float32'))
            manifest = root / 'signals.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'cells': [{
                'cell_id': 'synthetic_failed_prediction', 'dataset': 'NASA',
                'feature_schema': FEATURE_SCHEMAS['NASA'][0],
                'cycle_unit': FEATURE_SCHEMAS['NASA'][2],
                'protocol_description': 'synthetic failure fixture',
                'diagnostic_protocol_match': True, 'aging_strategy_relation': 'novel',
                'operating_conditions': {'charge_rate_c': None, 'discharge_rate_c': None,
                    'diagnostic_discharge_rate_c': None, 'ambient_temperature_c': None,
                    'strategy': 'synthetic failure fixture'},
                'signal_file': 'signal.npz'}]}))
            output = root / 'blind'
            with (patch('external_test_v2.verify'),
                  patch('external_test_v2.reject_training_overlap'),
                  patch('external_test_v2.predict_series',
                        side_effect=RuntimeError('synthetic inference failure'))):
                with self.assertRaisesRegex(RuntimeError, 'synthetic inference failure'):
                    predict(manifest, output)
            self.assertFalse(output.exists())
            staging = list(root.glob('.blind.staging-*'))
            self.assertEqual(len(staging), 1)
            self.assertFalse((staging[0] / 'prediction_manifest.json').exists())

    def test_signal_changed_during_prediction_is_not_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cycle = np.arange(1, 25)
            raw = np.ones((len(cycle), 44), dtype='float32')
            signal = root / 'signal.npz'
            np.savez_compressed(signal, cycle=cycle, raw_features=raw)
            manifest = root / 'signals.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'cells': [{
                'cell_id': 'synthetic_signal_mutation', 'dataset': 'NASA',
                'feature_schema': FEATURE_SCHEMAS['NASA'][0],
                'cycle_unit': FEATURE_SCHEMAS['NASA'][2],
                'protocol_description': 'synthetic mutation fixture',
                'diagnostic_protocol_match': True, 'aging_strategy_relation': 'novel',
                'operating_conditions': {'charge_rate_c': None, 'discharge_rate_c': None,
                    'diagnostic_discharge_rate_c': None, 'ambient_temperature_c': None,
                    'strategy': 'synthetic mutation fixture'},
                'signal_file': 'signal.npz'}]}))
            original_predict = external_test_v2.predict_series

            def mutate_signal(*args, **kwargs):
                result = original_predict(*args, **kwargs)
                np.savez_compressed(signal, cycle=cycle, raw_features=raw + 1)
                return result

            output = root / 'blind'
            with (patch('external_test_v2.verify'),
                  patch('external_test_v2.reject_training_overlap'),
                  patch('external_test_v2.predict_series', side_effect=mutate_signal)):
                with self.assertRaisesRegex(ValueError, 'signal differs from blind-stage hash'):
                    predict(manifest, output)
            self.assertFalse(output.exists())
            self.assertEqual(len(list(root.glob('.blind.staging-*'))), 1)

    def test_rul_scores_all_signal_cycles_when_capacity_labels_are_sparse(self):
        cycle = np.arange(1, 31)
        prediction = {'cycle': cycle, 'soh': np.full(30, .9),
                      'soh_band_lower': np.full(30, .8),
                      'soh_band_upper': np.ones(30),
                      'rul_cycles': np.full(30, 10.),
                      'rul_band_lower': np.zeros(30),
                      'rul_band_upper': np.full(30, 25.)}
        _, soh, rul = score_cell(prediction, np.array([1, 10, 20, 30]),
                                 np.array([2., 1.9, 1.8, 1.7]), 'NASA', 25)
        self.assertEqual(soh['observations'], 3)
        self.assertEqual(rul['observations'], 10)  # cycles 16 through 25

    def test_two_stage_smoke_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cycle = np.arange(1, 61)
            raw = np.zeros((60, FEATURE_SCHEMAS['NASA'][1]), dtype='float32')
            raw[:, :11] = np.linspace(4.1, 3.5, 11)[None, :] - .001*cycle[:, None]
            raw[:, 11:22] = -2.
            raw[:, 22:33] = 25.
            signal = root / 'signal.npz'
            np.savez_compressed(signal, raw_features=raw, cycle=cycle)
            signal_manifest = root / 'signals.json'
            signal_manifest.write_text(json.dumps({'schema_version': 1, 'cells': [{
                'cell_id': 'synthetic_smoke_cell', 'dataset': 'NASA',
                'feature_schema': FEATURE_SCHEMAS['NASA'][0],
                'cycle_unit': FEATURE_SCHEMAS['NASA'][2],
                'protocol_description': 'synthetic adapter smoke test only',
                'diagnostic_protocol_match': True,
                'aging_strategy_relation': 'novel',
                'future_strategy_relation': 'changed',
                'future_strategy_description': 'synthetic future policy change',
                'operating_conditions': {'charge_rate_c': None, 'discharge_rate_c': None,
                    'diagnostic_discharge_rate_c': None, 'ambient_temperature_c': None,
                    'strategy': 'synthetic smoke fixture'},
                'signal_file': 'signal.npz'}]}))
            blind = root / 'blind'
            predict(signal_manifest, blind)
            self.assertTrue((blind / 'prediction_manifest.json').exists())
            with np.load(blind / 'predictions/synthetic_smoke_cell.npz') as saved:
                self.assertEqual(len(saved['future_cycle']), 0)
            capacity_file = root / 'labels.npz'
            np.savez_compressed(capacity_file, cycle=cycle,
                                capacity_ah=np.linspace(2., 1.4, len(cycle)))
            label_manifest = root / 'labels.json'
            label_manifest.write_text(json.dumps({'schema_version': 1, 'capacity_unit': 'Ah',
                'cells': [{'cell_id': 'synthetic_smoke_cell', 'label_file': 'labels.npz',
                           'endpoint_rule': 'derived_80pct_sustained',
                           'complete_comparable_capacity_diagnostics': True,
                           'capacity_protocol_description': 'synthetic full diagnostic fixture'}]}))
            failed_score = root / 'failed_score.json'
            with (patch('external_test_v2.verify'),
                  patch('external_test_v2.json.dump',
                        side_effect=RuntimeError('synthetic score write failure'))):
                with self.assertRaisesRegex(RuntimeError, 'synthetic score write failure'):
                    score(blind, label_manifest, failed_score)
            self.assertFalse(failed_score.exists())
            self.assertEqual(len(list(root.glob('.failed_score.staging-*'))), 1)
            original_read_labels = external_test_v2.read_labels

            def mutate_labels(path, domain):
                observed_cycle, observed_capacity = original_read_labels(path, domain)
                np.savez_compressed(path, cycle=observed_cycle,
                                    capacity_ah=observed_capacity * .99)
                return observed_cycle, observed_capacity

            mutated_score = root / 'mutated_score.json'
            with (patch('external_test_v2.verify'),
                  patch('external_test_v2.read_labels', side_effect=mutate_labels)):
                with self.assertRaisesRegex(ValueError,
                                            'capacity label file changed while scoring'):
                    score(blind, label_manifest, mutated_score)
            self.assertFalse(mutated_score.exists())
            np.savez_compressed(capacity_file, cycle=cycle,
                                capacity_ah=np.linspace(2., 1.4, len(cycle)))
            output = root / 'first_score.json'
            score(blind, label_manifest, output)
            report = json.loads(output.read_text())
            self.assertEqual(report['stage'], 'first_frozen_external_score')
            self.assertEqual(report['domains']['NASA']['cells'], 1)
            self.assertEqual(report['domains']['NASA']['rul_observed_cells'], 1)
            self.assertIn('changed', report['domains']['NASA']['by_future_strategy_relation'])
            with self.assertRaises(FileExistsError):
                score(blind, label_manifest, output)

    def test_optional_xjtu_300s_challenger_is_scored_pairwise(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cycle = np.arange(1, 26)
            rng = np.random.default_rng(2026)
            raw120 = rng.normal(1., .01, (len(cycle), 46)).astype('float32')
            raw300 = rng.normal(1., .01, (len(cycle), 82)).astype('float32')
            np.savez_compressed(root / 'signal_120.npz', cycle=cycle, raw_features=raw120)
            np.savez_compressed(root / 'signal_300.npz', cycle=cycle, raw_features=raw300)
            signal_manifest = root / 'signals.json'
            signal_manifest.write_text(json.dumps({'schema_version': 1, 'cells': [{
                'cell_id': 'synthetic_xjtu_challenger', 'dataset': 'XJTU',
                'feature_schema': FEATURE_SCHEMAS['XJTU'][0],
                'cycle_unit': FEATURE_SCHEMAS['XJTU'][2],
                'protocol_description': 'synthetic smoke fixture only',
                'diagnostic_protocol_match': True,
                'aging_strategy_relation': 'novel',
                'future_strategy_relation': 'same',
                'operating_conditions': {'charge_rate_c': None,
                    'discharge_rate_c': None, 'diagnostic_discharge_rate_c': 1.,
                    'ambient_temperature_c': None, 'strategy': 'synthetic smoke fixture'},
                'signal_file': 'signal_120.npz',
                'challenger_signal_file': 'signal_300.npz',
                'optional_challengers': ['xjtu_raw_cycle_rul_v1']}]}))
            blind = root / 'blind'
            predict(signal_manifest, blind)
            with np.load(blind / 'predictions/synthetic_xjtu_challenger.npz') as saved:
                self.assertIn('soh_challenger_300', saved.files)
                self.assertTrue(np.isnan(saved['soh_challenger_300'][:5]).all())
                self.assertIn('rul_challenger_raw_target', saved.files)
                self.assertTrue(np.isnan(saved['rul_challenger_raw_target'][:15]).all())
            np.savez_compressed(root / 'labels.npz', cycle=cycle,
                                capacity_ah=np.linspace(2., 1.6, len(cycle)))
            label_manifest = root / 'labels.json'
            label_manifest.write_text(json.dumps({'schema_version': 1,
                'capacity_unit': 'Ah', 'cells': [{
                    'cell_id': 'synthetic_xjtu_challenger', 'label_file': 'labels.npz',
                    'endpoint_rule': 'xjtu_source_cycle_life', 'endpoint_observed': True,
                    'endpoint_cycle': 25,
                    'complete_comparable_capacity_diagnostics': True,
                    'capacity_protocol_description': 'synthetic smoke fixture'}]}))
            output = root / 'score.json'
            score(blind, label_manifest, output)
            report = json.loads(output.read_text())
            self.assertEqual(report['domains']['XJTU']['paired_300s_challenger']['cells'], 1)
            self.assertEqual(report['domains']['XJTU']['paired_300s_challenger']
                             ['cell_bootstrap_status'], 'INSUFFICIENT_CELLS')
            self.assertIn('soh_challenger_300', report['cells'][0])
            self.assertEqual(report['domains']['XJTU']
                             ['paired_raw_target_rul_challenger']['cells'], 1)
            self.assertEqual(report['domains']['XJTU']
                             ['paired_raw_target_rul_challenger']
                             ['cell_bootstrap_status'], 'INSUFFICIENT_CELLS')
            self.assertIn('rul_challenger_raw_target', report['cells'][0])
            self.assertIn('by_feature_applicability', report['domains']['XJTU'])
            self.assertIn('same', report['domains']['XJTU']['by_future_strategy_relation'])
            self.assertIn('NOT VALIDATED', json.loads((blind / 'prediction_manifest.json')
                                                  .read_text())['cells'][0]['rul_reliability_status'])


if __name__ == '__main__':
    unittest.main()
