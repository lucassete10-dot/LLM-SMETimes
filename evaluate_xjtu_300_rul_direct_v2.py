"""Development ablation: XJTU direct RUL with 300-second signal features.

Uses the frozen direct route's labels, weights, downsampling, estimator and
whole-strategy splits. No assisted SOH or target capacity enters features.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from evaluate_xjtu_long_prefix_v2 import ROOT, matrix
from train_rul_v2 import assess, eligible
from train_soh_v2 import outer_splits


BASELINE = Path('battery_models_v2/rul/xjtu/group_cv.json')
OUT = ROOT / 'direct_rul_300s_diagnostic.json'


def append_age(feature, cycle):
    age = np.asarray(cycle, dtype='float32')
    return np.column_stack((feature, age, np.log1p(age))).astype('float32')


def fit(cells, features, ids):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        keep = eligible(cell)
        if not keep.any():
            continue
        x = features[key][keep]
        y = cell.rul_cycles[keep]
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        weight /= weight.sum()
        x_parts.append(x[::5])
        y_parts.append(y[::5])
        w_parts.append(weight[::5] / len(ids))
    x, y, weight = map(np.concatenate, (x_parts, y_parts, w_parts))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), np.log1p(y), sample_weight=weight)
    return imputer, model


def evaluate():
    manifest, cells = load_all()
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    prior = {row['held_out']: row['direct'] for row in json.loads(
        BASELINE.read_text(encoding='utf-8'))}
    features = {}
    with np.load(ROOT / 'features.npz') as archive:
        for row in records:
            key = row['cell_id']
            if not np.array_equal(archive[f'{key}_cycle'], cells[key].cycle):
                raise ValueError(f'{key}: 300s feature/cycle mismatch')
            raw = archive[f'{key}_300s'].astype('float32')
            features[key] = append_age(matrix(raw, cells[key].reference_count), cells[key].cycle)
    folds = []
    for group, train_ids, val_ids in outer_splits(manifest['cells'], 'XJTU'):
        imputer, model = fit(cells, features, train_ids)
        predictions = {key: np.maximum(0, np.expm1(model.predict(
            imputer.transform(features[key])))) for key in val_ids}
        baseline_life = float(np.median([cells[key].endpoint_cycle for key in train_ids]))
        candidate = assess(cells, val_ids, predictions, baseline_life)
        folds.append({'held_out': group, '120s_direct': prior[group], '300s_direct': candidate})
        print(group, '120s', round(prior[group]['cell_macro_mae_cycles'], 1),
              '300s', round(candidate['cell_macro_mae_cycles'], 1), flush=True)
    summary = {name: {
        'equal_fold_macro_mae_cycles': float(np.mean(
            [row[name]['cell_macro_mae_cycles'] for row in folds])),
        'worst_cell_mae_cycles': float(max(cell['mae_cycles'] for row in folds
                                           for cell in row[name]['by_cell'])),
        'improved_folds_vs_120s': sum(row[name]['cell_macro_mae_cycles'] <
                                      row['120s_direct']['cell_macro_mae_cycles']
                                      for row in folds),
    } for name in ('120s_direct', '300s_direct')}
    return {'status': 'development-only 300s direct-RUL ablation; frozen RUL unchanged',
            'selection_warning': '300s window was proposed after development SOH analysis',
            'summary': summary, 'folds': folds}


if __name__ == '__main__':
    result = evaluate()
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', OUT, json.dumps(result['summary'], indent=2), flush=True)
