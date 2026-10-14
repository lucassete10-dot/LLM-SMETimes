"""Whole-strategy-held-out XJTU 300/600-second SOH feature comparison.

Estimator, weights, and outer folds match the frozen 120-second ExtraTrees
baseline. This development experiment does not replace the frozen candidate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import derive_signal_features, load_all
from train_soh_v2 import outer_splits, score


ROOT = Path('battery_models_v2/xjtu_long_prefix')
BASELINE = Path('battery_models_v2/soh/xjtu/group_cv.json')


def matrix(raw, reference_count):
    delta, history = derive_signal_features(raw, reference_count)
    return np.concatenate((raw, delta, history), axis=1)


def fit(cells, features, ids):
    inputs, labels, weights = [], [], []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        inputs.append(features[key][keep])
        labels.append(cell.soh[keep])
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        weights.append(weight / weight.sum() / len(ids))
    x = np.concatenate(inputs)
    y = np.concatenate(labels)
    w = np.concatenate(weights)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=w)
    return imputer, model


def evaluate(windows):
    manifest, cells = load_all()
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    prior = {fold['held_out']: fold['extra_trees_history'] for fold in
             json.loads(BASELINE.read_text(encoding='utf-8'))}
    with np.load(ROOT / 'features.npz') as archive:
        features = {}
        for window in windows:
            by_cell = {}
            for row in records:
                key = row['cell_id']
                cycle = archive[f'{key}_cycle']
                raw = archive[f'{key}_{window}s'].astype('float32')
                if not np.array_equal(cycle, cells[key].cycle):
                    raise ValueError(f'{key}: source cycle mismatch')
                by_cell[key] = matrix(raw, cells[key].reference_count)
            features[window] = by_cell
    folds = []
    for group, train_ids, val_ids in outer_splits(manifest['cells'], 'XJTU'):
        result = {'held_out': group, '120s_baseline': prior[group]}
        for window in windows:
            imputer, model = fit(cells, features[window], train_ids)
            predictions = {key: model.predict(imputer.transform(features[window][key]))
                           for key in val_ids}
            result[f'{window}s'] = score(cells, predictions, val_ids)
        folds.append(result)
        print(group, {name: round(result[name]['cell_macro_mae'], 5)
                      for name in ('120s_baseline', *(f'{window}s' for window in windows))},
              flush=True)
    summary = {}
    for name in ('120s_baseline', *(f'{window}s' for window in windows)):
        summary[name] = {
            'equal_fold_macro_mae': float(np.mean([fold[name]['cell_macro_mae'] for fold in folds])),
            'worst_cell_mae': float(max(row['mae'] for fold in folds
                                        for row in fold[name]['by_cell'])),
            'improved_folds_vs_120s': sum(
                fold[name]['cell_macro_mae'] < fold['120s_baseline']['cell_macro_mae']
                for fold in folds),
        }
    return {
        'status': 'development-only, post-hoc long-prefix experiment; frozen model unchanged',
        'selection_warning': 'window lengths were proposed after examining development Batch 5 errors; outer folds are not independent final tests',
        'evaluation': 'same 55 cells, 10389 diagnostics, sample weights, ExtraTrees parameters, and whole-strategy folds',
        'summary': summary, 'folds': folds,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', nargs='+', type=int, choices=(300, 600),
                        default=[300, 600])
    args = parser.parse_args()
    result = evaluate(args.windows)
    (ROOT / 'group_cv.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', ROOT / 'group_cv.json', json.dumps(result['summary'], indent=2), flush=True)
