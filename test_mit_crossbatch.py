"""Checks that saved cross-batch scores match untouched held-out predictions."""
import csv
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from battery_mvp import compute_metrics
from battery_mit_crossbatch import CHECKPOINTS, OUTPUT, load_batch


class CrossBatchTests(unittest.TestCase):
    def test_no_continuation_or_author_bad_cell(self):
        b2, _, _ = load_batch(2)
        b3, _, _ = load_batch(3)
        self.assertEqual(len(b2), 43)
        self.assertEqual(len(b3), 45)
        self.assertFalse(set(b2) & {'b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16'})
        self.assertNotIn('b3c37', b3)

    def test_frozen_checkpoint_only_names_batch1(self):
        for name in ('common_numeric_2026.pt', 'enhanced_robust_2026.pt'):
            saved = torch.load(CHECKPOINTS / name, map_location='cpu', weights_only=False)
            self.assertEqual(len(saved['train_ids']), 28)
            self.assertTrue(all(key.startswith('b1c') for key in saved['train_ids']))
            self.assertTrue(saved['val_id'].startswith('b1c'))

    def test_recomputed_macro_mae(self):
        for batch, expected_cells, expected_rows in [(2, 43, 20868), (3, 45, 48943)]:
            folder = OUTPUT / f'batch{batch}'
            published = json.loads((folder / 'metrics.json').read_text())
            with (folder / 'predictions.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), expected_rows)
            self.assertEqual(len({row['cell'] for row in rows}), expected_cells)
            grouped = {}
            for row in rows:
                grouped.setdefault(row['cell'], []).append(row)
            for name in published['macro']:
                scores = []
                for subset in grouped.values():
                    pred = np.asarray([float(r[name]) for r in subset])
                    actual = np.asarray([float(r['true_soh']) for r in subset])
                    scores.append(compute_metrics(pred, actual)['mae'])
                self.assertAlmostEqual(float(np.mean(scores)), published['macro'][name]['mae'], places=9)


if __name__ == '__main__':
    unittest.main()
