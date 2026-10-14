"""MIT whole-batch online SOH with robust, regularized boosting.

Uses the same current/past 300-second signal and observed cycle age as the
age-augmented ExtraTrees benchmark. No measured validation capacity, SOH,
future cycle, endpoint, or batch identity enters an inference row.
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from battery_v2_features import load_cell, load_manifest
from evaluate_soh_age_signal_v2 import age_matrix
from train_soh_v2 import outer_splits


ROOT = Path('battery_models_v2/soh_age_signal_challenger')
BASE = ROOT / 'mit_development_oof_predictions.npz'
OUT = ROOT / 'mit_histgb_development_cv.json'
OOF = ROOT / 'mit_histgb_development_oof.npz'
SETTINGS = dict(loss='absolute_error', learning_rate=.05, max_iter=120,
                max_leaf_nodes=15, min_samples_leaf=60,
                l2_regularization=10., max_bins=63,
                early_stopping=False, random_state=2026)


def fit(cells, matrices, training):
    x_parts, y_parts, w_parts = [], [], []
    for key in training:
        cell = cells[key]
        keep = cell.soh_available
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype(float)
        weight /= weight.sum()*len(training)
        x_parts.append(matrices[key][keep])
        y_parts.append(cell.soh[keep])
        w_parts.append(weight)
    x, y, weight = map(np.concatenate, (x_parts, y_parts, w_parts))
    model = HistGradientBoostingRegressor(**SETTINGS)
    model.fit(x, y, sample_weight=weight*len(weight))
    return model


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    matrices = {key: age_matrix(cell) for key, cell in cells.items()}
    with np.load(BASE) as archive:
        baseline = {(str(key), int(cycle)): float(estimate)
                    for key, cycle, estimate in zip(archive['cell_id'],
                        archive['cycle'], archive['gated_with_age'])}
    folds = []
    heldout = {}
    for group, training, validation in outer_splits(records, 'MIT'):
        start = time.perf_counter()
        model = fit(cells, matrices, training)
        fit_seconds = time.perf_counter()-start
        scored = []
        for key in validation:
            cell = cells[key]
            candidate = model.predict(matrices[key])
            heldout[key] = candidate
            original = np.array([baseline[key, int(cycle)]
                                 for cycle in cell.cycle])
            keep = cell.soh_available
            target = cell.soh[keep]
            scored.append({
                'cell_id': key,
                'baseline_mape_percent': float(100*np.mean(np.abs(
                    original[keep]-target)/target)),
                'candidate_mape_percent': float(100*np.mean(np.abs(
                    candidate[keep]-target)/target)),
            })
        folds.append({
            'held_out_batch': group, 'training_cells': len(training),
            'scored_cells': len(scored), 'fit_seconds': fit_seconds,
            'baseline_mape_percent': float(np.mean([
                row['baseline_mape_percent'] for row in scored])),
            'candidate_mape_percent': float(np.mean([
                row['candidate_mape_percent'] for row in scored])),
            'by_cell': scored,
        })
        print(group, 'baseline', folds[-1]['baseline_mape_percent'],
              'candidate', folds[-1]['candidate_mape_percent'],
              'fit seconds', round(fit_seconds, 1), flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    report = {
        'status': 'unpromoted MIT whole-batch robust boosting SOH ablation',
        'input': 'current and previous 300 s discharge prefixes plus observed cycle number',
        'settings': SETTINGS,
        'summary': {'scored_cells': len(rows),
                    'baseline_mape_percent': float(np.mean([
                        row['baseline_mape_percent'] for row in rows])),
                    'candidate_mape_percent': float(np.mean([
                        row['candidate_mape_percent'] for row in rows]))},
        'folds': folds,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    ids = [row['cell_id'] for row in records]
    np.savez_compressed(OOF,
                        cell_id=np.concatenate([np.full(len(cells[key].cycle), key)
                                                for key in ids]),
                        cycle=np.concatenate([cells[key].cycle for key in ids]),
                        target=np.concatenate([cells[key].soh for key in ids]),
                        soh_available=np.concatenate([cells[key].soh_available for key in ids]),
                        estimate=np.concatenate([heldout[key] for key in ids]))
    print('SUMMARY', report['summary'])
    print('WROTE', OUT, OOF)


if __name__ == '__main__':
    main()
