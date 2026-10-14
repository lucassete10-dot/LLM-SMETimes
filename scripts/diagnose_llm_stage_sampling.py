"""Audit TRAIN-only stage supervision distortion from capped target sampling."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json, sha


def stage_drops(x, y, robust=False):
    x, y = np.asarray(x), np.asarray(y)
    span = x[-1] - x[0]
    result = []
    for j in range(3):
        mask = (x >= x[0] + j * span / 3) & (x <= x[0] + (j + 1) * span / 3)
        xx, yy = x[mask], y[mask]
        if len(xx) < 2:
            raise ValueError('insufficient stage supervision')
        width = max(1., min(5., (xx[-1] - xx[0]) / 4))
        reduction = np.median if robust else np.mean
        result.append(reduction(yy[xx <= xx[0] + width]) - reduction(yy[xx >= xx[-1] - width]))
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.run)
    cfg = json.loads((root / 'config.json').read_text())
    audit = Path(cfg['frozen_plan'])
    frozen = json.loads((audit / 'config.json').read_text())
    prepared = json.loads((audit / 'audit.json').read_text())['prepared_folder']
    _, cells = load_data(Path(prepared), frozen['config']['integrity_audit'])
    rows = []
    for cid in cfg['plan']['train']:
        cell = cells[cid]
        take = np.flatnonzero((cell['label_cycle'] > cfg['origin']) & np.isfinite(cell['soh']) & (cell['soh'] > 0))
        x, y = cell['label_cycle'][take], cell['soh'][take]
        if len(x) < 9:
            continue
        sample = np.unique(np.linspace(0, len(x) - 1, cfg['future_targets']).round().astype(int)) if len(x) > cfg['future_targets'] else np.arange(len(x))
        dense = stage_drops(x, y, robust=True)
        sampled = stage_drops(x[sample], y[sample])
        dense_mean = stage_drops(x, y)
        rows.append({'cell': cid, 'full_points': len(x), 'sampled_points': len(sample),
                     'dense_robust_stage_drop_pp': (100 * dense).tolist(),
                     'sampled_training_stage_drop_pp': (100 * sampled).tolist(),
                     'max_sampling_target_difference_pp': float(100 * np.max(abs(sampled - dense))),
                     'max_dense_mean_vs_median_pp': float(100 * np.max(abs(dense_mean - dense)))})
    error = np.asarray([r['max_sampling_target_difference_pp'] for r in rows])
    result = {'scope': 'predefined TRAIN labels only; diagnostic, not forecasting or heldout selection',
              'run_config_sha256': sha(root / 'config.json'), 'origin': cfg['origin'],
              'summary': {'cells': len(rows), 'mean_max_difference_pp': float(error.mean()),
                          'median_max_difference_pp': float(np.median(error)),
                          'worst_difference_pp': float(error.max()), 'cells_above_2pp': int((error > 2).sum())},
              'by_cell': rows}
    save_json(root / 'TRAIN_stage_sampling_diagnostic.json', result)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
