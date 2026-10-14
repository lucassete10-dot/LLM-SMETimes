"""Leakage-controlled XJTU assisted-RUL ablation with 300-second SOH signals.

For each held-out strategy, training RUL receives only SOH estimates from
inner models trained on other strategies. The outer validation strategy is
excluded from every fitted component. This remains a development experiment.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from evaluate_xjtu_long_prefix_v2 import ROOT, fit as fit_soh, matrix
from train_rul_v2 import assess, eligible
from train_soh_v2 import outer_splits


OUT = ROOT / 'assisted_rul_300s_diagnostic.json'
BASELINE = Path('battery_models_v2/rul/xjtu/group_cv.json')
DIRECT_300 = ROOT / 'direct_rul_300s_diagnostic.json'


def predict_soh(model, features):
    imputer, regressor = model
    return regressor.predict(imputer.transform(features)).astype('float32')


def assisted_matrix(signal, cycle, soh_estimate):
    age = np.asarray(cycle, dtype='float32')
    soh_estimate = np.asarray(soh_estimate, dtype='float32')
    if len(signal) != len(age) or len(signal) != len(soh_estimate):
        raise ValueError('assisted RUL input alignment mismatch')
    base = np.column_stack((signal, age, np.log1p(age)))
    index = np.arange(len(age))
    lag_index = np.maximum(0, index-20)
    lag = soh_estimate[lag_index]
    slope = (soh_estimate-lag) / np.maximum(1, index-lag_index)
    smooth = np.asarray([np.median(soh_estimate[max(0, i-4):i+1])
                         for i in index], dtype='float32')
    return np.column_stack((base, soh_estimate, smooth, slope))


def crossfit_soh(cells, records, features, train_ids):
    by_group = {}
    for row in records:
        if row['cell_id'] in train_ids:
            by_group.setdefault(row['group'], []).append(row['cell_id'])
    if len(by_group) < 2:
        raise ValueError('inner SOH cross-fitting requires at least two strategies')
    predictions = {}
    for group, held_out in by_group.items():
        fit_ids = [key for key in train_ids if key not in held_out]
        model = fit_soh(cells, features, fit_ids)
        predictions.update({key: predict_soh(model, features[key]) for key in held_out})
        print('  INNER', group, len(held_out), flush=True)
    if set(predictions) != set(train_ids):
        raise ValueError('incomplete inner SOH predictions')
    return predictions


def fit_assisted_rul(cells, signals, ids, soh_predictions):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        keep = eligible(cell)
        x = assisted_matrix(signals[key], cell.cycle, soh_predictions[key])[keep]
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
    ids = [row['cell_id'] for row in records]
    signals = {}
    with np.load(ROOT / 'features.npz') as archive:
        for key in ids:
            if not np.array_equal(archive[f'{key}_cycle'], cells[key].cycle):
                raise ValueError(f'{key}: feature/cycle mismatch')
            signals[key] = matrix(archive[f'{key}_300s'].astype('float32'),
                                  cells[key].reference_count)
    old = {row['held_out']: row for row in json.loads(BASELINE.read_text(encoding='utf-8'))}
    direct_300 = {row['held_out']: row['300s_direct'] for row in json.loads(
        DIRECT_300.read_text(encoding='utf-8'))['folds']}
    folds = []
    for group, train_ids, val_ids in outer_splits(manifest['cells'], 'XJTU'):
        print('OUTER', group, 'train', len(train_ids), 'validation', len(val_ids), flush=True)
        inner_soh = crossfit_soh(cells, records, signals, train_ids)
        rul_imputer, rul_model = fit_assisted_rul(cells, signals, train_ids, inner_soh)
        outer_soh_model = fit_soh(cells, signals, train_ids)
        estimates = {}
        for key in val_ids:
            soh = predict_soh(outer_soh_model, signals[key])
            x = assisted_matrix(signals[key], cells[key].cycle, soh)
            estimates[key] = np.maximum(0, np.expm1(
                rul_model.predict(rul_imputer.transform(x))))
        baseline_life = float(np.median([cells[key].endpoint_cycle for key in train_ids]))
        candidate = assess(cells, val_ids, estimates, baseline_life)
        folds.append({'held_out': group,
                      '120s_direct': old[group]['direct'],
                      '120s_assisted': old[group]['assisted'],
                      '300s_direct': direct_300[group],
                      '300s_assisted': candidate})
        print('SCORED', group, round(candidate['cell_macro_mae_cycles'], 1), flush=True)
    summary = {name: {
        'equal_fold_macro_mae_cycles': float(np.mean(
            [row[name]['cell_macro_mae_cycles'] for row in folds])),
        'worst_cell_mae_cycles': float(max(cell['mae_cycles'] for row in folds
                                           for cell in row[name]['by_cell'])),
    } for name in ('120s_direct', '120s_assisted', '300s_direct', '300s_assisted')}
    return {'status': 'post-hoc development-only; frozen RUL unchanged',
            'soh_crossfit': 'inner whole-strategy models fitted within each outer training fold',
            'summary': summary, 'folds': folds}


if __name__ == '__main__':
    result = evaluate()
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', OUT, json.dumps(result['summary'], indent=2), flush=True)
