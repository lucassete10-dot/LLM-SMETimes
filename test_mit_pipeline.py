"""Lightweight regression checks for MIT preprocessing and held-out results."""
import csv
import json
import unittest
from pathlib import Path

import numpy as np

from battery_mvp import compute_metrics
from prepare_mit import COMMON_GRID, GRID, COMMON_NAMES, ENHANCED_NAMES, feature_vector


ROOT = Path('battery_mit_results')


class MITPipelineTests(unittest.TestCase):
    def test_prefix_does_not_read_future_measurements(self):
        t = np.arange(0, 361, 30, dtype=float) / 60
        i = np.full(len(t), -4.)
        v = 3.8 - np.arange(len(t)) * .02
        temp = np.full(len(t), 29.)
        for grid in (COMMON_GRID, GRID):
            before = feature_vector(t, i, v, temp, grid)
            after_v, after_t = v.copy(), temp.copy()
            after_v[t * 60 > 300] = -99
            after_t[t * 60 > 300] = 999
            np.testing.assert_allclose(before, feature_vector(t, i, after_v, after_t, grid), equal_nan=True)

    def test_feature_schema_has_no_target(self):
        self.assertEqual(len(COMMON_NAMES), 15)
        self.assertEqual(len(ENHANCED_NAMES), 51)
        for name in COMMON_NAMES + ENHANCED_NAMES:
            self.assertNotIn('capacity', name)
            self.assertNotIn('soh', name)
            self.assertNotIn('discharge_duration', name)

    def test_cell_split_is_disjoint(self):
        split = json.loads((ROOT / 'mit_cell_holdout/split.json').read_text())
        groups = [set(split[key]) for key in ('train', 'validation', 'test')]
        self.assertEqual([len(g) for g in groups], [28, 8, 10])
        self.assertEqual(len(set.union(*groups)), 46)
        self.assertEqual(sum(map(len, groups)), 46)

    def test_saved_metrics_recalculate(self):
        folder = ROOT / 'mit_cell_holdout'
        published = json.loads((folder / 'metrics.json').read_text())
        with (folder / 'predictions.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8372)
        for model in published['macro']:
            cell_scores = []
            for cell in published['by_cell']:
                subset = [r for r in rows if r['cell'] == cell]
                result = compute_metrics(np.array([float(r[model]) for r in subset]),
                                         np.array([float(r['true_soh']) for r in subset]))
                cell_scores.append(result['mae'])
            self.assertAlmostEqual(float(np.mean(cell_scores)), published['macro'][model]['mae'], places=9)

    def test_original_audit(self):
        manifest = json.loads(Path(r'D:\LLM\MIT\processed\manifest.json').read_text())
        self.assertEqual(manifest['saved_cells'], 46)
        self.assertEqual(manifest['valid_cycles'], 38747)
        self.assertEqual(manifest['excluded_reasons']['capacity_above_physical_screen_1p25Ah'], 2)


if __name__ == '__main__':
    unittest.main()
