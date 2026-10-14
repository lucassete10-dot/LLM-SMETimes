"""Whole-strategy XJTU direct-RUL ablation with causal historical usage signals.

Compares frozen 120s and development 300s diagnostic features, with/without
prior-cycle current/temperature summaries. No future controls are provided;
the interpretation assumes the observed usage distribution continues.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from evaluate_xjtu_long_prefix_v2 import ROOT as LONG_ROOT, matrix
from train_rul_v2 import assess, eligible, signal_features
from train_soh_v2 import outer_splits


ROOT = Path('battery_models_v2/xjtu_usage_history')
OUT = ROOT / 'direct_rul_group_cv.json'
BASELINE = Path('battery_models_v2/rul/xjtu/group_cv.json')
DIRECT_300 = LONG_ROOT / 'direct_rul_300s_diagnostic.json'


def fit(cells, features, ids):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        keep = eligible(cell)
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
    old = {row['held_out']: row['direct'] for row in json.loads(
        BASELINE.read_text(encoding='utf-8'))}
    direct300 = {row['held_out']: row['300s_direct'] for row in json.loads(
        DIRECT_300.read_text(encoding='utf-8'))['folds']}
    by_variant = {'120s_plus_usage': {}, '300s_plus_usage': {}}
    with np.load(ROOT / 'features.npz') as usage_file, \
            np.load(LONG_ROOT / 'features.npz') as prefix_file:
        for row in records:
            key = row['cell_id']
            cell = cells[key]
            if (not np.array_equal(usage_file[f'{key}_cycle'], cell.cycle) or
                    not np.array_equal(prefix_file[f'{key}_cycle'], cell.cycle)):
                raise ValueError(f'{key}: usage/diagnostic cycle alignment mismatch')
            usage = usage_file[f'{key}_usage'].astype('float32')
            if usage.shape != (len(cell.cycle), 16):
                raise ValueError(f'{key}: historical usage shape mismatch')
            baseline = signal_features(cell)
            raw300 = prefix_file[f'{key}_300s'].astype('float32')
            diagnostic300 = matrix(raw300, cell.reference_count)
            age = cell.cycle.astype('float32')
            longer = np.column_stack((diagnostic300, age, np.log1p(age))).astype('float32')
            by_variant['120s_plus_usage'][key] = np.column_stack((baseline, usage))
            by_variant['300s_plus_usage'][key] = np.column_stack((longer, usage))
    folds = []
    for group, train_ids, val_ids in outer_splits(manifest['cells'], 'XJTU'):
        base_life = float(np.median([cells[key].endpoint_cycle for key in train_ids]))
        result = {'held_out': group, '120s_direct': old[group],
                  '300s_direct': direct300[group]}
        for variant, features in by_variant.items():
            imputer, model = fit(cells, features, train_ids)
            predictions = {key: np.maximum(0, np.expm1(model.predict(
                imputer.transform(features[key])))) for key in val_ids}
            result[variant] = assess(cells, val_ids, predictions, base_life)
        folds.append(result)
        print(group, {name: round(result[name]['cell_macro_mae_cycles'], 1)
                      for name in ('120s_direct', '120s_plus_usage',
                                   '300s_direct', '300s_plus_usage')}, flush=True)
    summary = {}
    for name in ('120s_direct', '120s_plus_usage', '300s_direct', '300s_plus_usage'):
        summary[name] = {
            'equal_fold_macro_mae_cycles': float(np.mean(
                [row[name]['cell_macro_mae_cycles'] for row in folds])),
            'worst_cell_mae_cycles': float(max(cell['mae_cycles'] for row in folds
                                               for cell in row[name]['by_cell'])),
            'improved_folds_vs_same_prefix_without_usage': sum(
                row[name]['cell_macro_mae_cycles'] < row[
                    '120s_direct' if name.startswith('120s') else '300s_direct'
                ]['cell_macro_mae_cycles'] for row in folds),
        }
    return {'status': 'development-only usage ablation; frozen candidate unchanged',
            'future_assumption': 'historical usage distribution continues; no future schedule input',
            'input': 'prior source cycles only; first 120s current and temperature; no capacity/throughput/duration',
            'summary': summary, 'folds': folds}


if __name__ == '__main__':
    result = evaluate()
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', OUT, json.dumps(result['summary'], indent=2), flush=True)
