import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from scripts.compare_llm_input_budgets import aligned_curves, fixed_horizons, verify_signal_extension
from scripts.prepare_same_type_forecast import sha, save_json


class BudgetComparisonTests(unittest.TestCase):
    def test_common_source_interval_not_row_count(self):
        x1 = np.array([50., 100., 103., 110., 150.])
        x2 = np.array([100., 103., 110., 150., 160.])
        y1 = 1-x1/1000
        y2 = 1-x2/1000
        x, y, a, b = aligned_curves((x1, y1, y1+.01), (x2, y2, y2+.02), 100)
        np.testing.assert_array_equal(x, [103, 110, 150])
        np.testing.assert_allclose(a-y, .01)
        np.testing.assert_allclose(b-y, .02)

    def test_disagreeing_truth_rejected(self):
        x = np.array([101., 110., 150.])
        y = np.ones(3)
        with self.assertRaises(AssertionError):
            aligned_curves((x, y, y), (x, y+.01, y), 100)

    def test_duplicate_cycles_rejected(self):
        x = np.array([101., 101., 150.])
        y = np.ones(3)
        with self.assertRaises(ValueError):
            aligned_curves((x, y, y), (x, y, y), 100)

    def test_incomplete_horizon_not_claimed_full(self):
        x = np.arange(51., 121.)
        y = np.ones(len(x))
        result = fixed_horizons((x, y, y-.01), 50)
        self.assertTrue(result['50']['coverage_reaches_requested_end'])
        self.assertFalse(result['100']['coverage_reaches_requested_end'])
        self.assertAlmostEqual(result['100']['metrics']['mae_pp'], 1.)

    def test_signal_extension_requires_identical_labels_and_old_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plans = [root/'parent', root/'extended']
            data = [root/'old_data', root/'new_data']
            for p in plans:
                (p/'source_snapshot').mkdir(parents=True)
                for name in ('splits.json', 'cell_registry.json', 'source_snapshot/battery_physical_registry.json'):
                    save_json(p/name, {'fixed': True})
            for p in data:
                p.mkdir()
            save_json(plans[0]/'config.json', {'frozen': True})
            save_json(plans[1]/'config.json', {
                'signal_extension_parent_config_sha256': sha(plans[0]/'config.json')})
            save_json(plans[1]/'signal_extension_binding.json', {'parent_plan': str(plans[0])})
            for p, d in zip(plans, data):
                save_json(p/'audit.json', {'prepared_folder': str(d)})
            cfg = [{'frozen_plan': str(p)} for p in plans]

            def prepare(change_label=False, change_prefix=False):
                provenance = []
                for i, d in enumerate(data):
                    count = 100 if i == 0 else 150
                    raw = np.ones((count, 2), dtype=np.float32)
                    labels = np.array([1., .9])
                    if i and change_label:
                        labels[-1] -= .01
                    if i and change_prefix:
                        raw[50, 0] += 1
                    file = d/'b1c0.npz'
                    np.savez(file, cycle=np.arange(1, count+1), raw_features=raw,
                             label_cycle=np.array([100, 200]), soh=labels)
                    digest = sha(file)
                    save_json(d/'manifest.json', {'cells': [{'cell_id': 'b1c0',
                        'metadata': {'dataset': 'MIT'}, 'file': str(file), 'sha256': digest}]})
                    provenance.append({'data_manifest_sha256': sha(d/'manifest.json'),
                                       'prepared_cells_sha256': {'b1c0': digest}})
                return provenance

            verify_signal_extension(cfg, prepare())
            with self.assertRaises(AssertionError):
                verify_signal_extension(cfg, prepare(change_label=True))
            with self.assertRaises(AssertionError):
                verify_signal_extension(cfg, prepare(change_prefix=True))
            provenance = prepare()
            save_json(plans[1]/'splits.json', {'changed': True})
            with self.assertRaises(ValueError):
                verify_signal_extension(cfg, provenance)
