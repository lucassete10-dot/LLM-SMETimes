"""Fixed-weight XJTU SOH ablation: equal training mass per strategy.

This leaves the selected 120-second signal-history model untouched. Every
outer validation strategy is excluded from imputation, fitting and weighting.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_cell, load_manifest
from train_soh_v2 import outer_splits, score


ROOT = Path('battery_models_v2')
OUT = ROOT / 'xjtu_group_balance'


def group_balanced_arrays(cells, ids):
    """Return training-only arrays with equal mass per group and per cell."""
    groups = Counter(cells[key].group for key in ids)
    if not ids or len(groups) < 2:
        raise ValueError('group balancing needs at least two training strategies')
    parts_x, parts_y, parts_w = [], [], []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        if not keep.any():
            raise ValueError(f'{key}: no scored SOH observations')
        review = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        review *= 1. / (review.sum() * len(groups) * groups[cell.group])
        parts_x.append(cell.matrix('signal_history')[keep])
        parts_y.append(cell.soh[keep])
        parts_w.append(review)
    x, y, weights = map(np.concatenate, (parts_x, parts_y, parts_w))
    if not np.isclose(weights.sum(), 1., atol=1e-12):
        raise ValueError('training weights do not sum to one')
    return x, y, weights


def fit(cells, ids):
    x, y, weights = group_balanced_arrays(cells, ids)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weights)
    return imputer, model


def evaluate():
    manifest = load_manifest()
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    cells = {row['cell_id']: load_cell(row) for row in records}
    baseline = {row['held_out']: row['extra_trees_history'] for row in json.loads(
        (ROOT / 'soh/xjtu/group_cv.json').read_text(encoding='utf-8'))}
    folds = []
    for group, train_ids, val_ids in outer_splits(records, 'XJTU'):
        if set(train_ids) & set(val_ids) or set(val_ids) != {
                key for key, cell in cells.items() if cell.group == group}:
            raise ValueError(f'{group}: invalid whole-strategy split')
        imputer, model = fit(cells, train_ids)
        predictions = {key: model.predict(imputer.transform(
            cells[key].matrix('signal_history'))) for key in val_ids}
        metric = score(cells, predictions, val_ids)
        if {row['cell_id'] for row in metric['by_cell']} != set(val_ids):
            raise ValueError(f'{group}: incomplete validation metrics')
        folds.append({'held_out': group, 'train_cell_ids': train_ids,
                      'validation_cell_ids': val_ids,
                      'baseline_extra_trees_history': baseline[group],
                      'equal_strategy_weight_extra_trees_history': metric})
        print(group, round(baseline[group]['cell_macro_mae'], 5),
              round(metric['cell_macro_mae'], 5), flush=True)
    candidates = ('baseline_extra_trees_history',
                  'equal_strategy_weight_extra_trees_history')
    summary = {name: {
        'equal_fold_macro_mae': float(np.mean([fold[name]['cell_macro_mae'] for fold in folds])),
        'worst_cell_mae': float(max(row['mae'] for fold in folds
                                    for row in fold[name]['by_cell'])),
    } for name in candidates}
    summary['equal_strategy_weight_extra_trees_history']['improved_folds'] = sum(
        fold['equal_strategy_weight_extra_trees_history']['cell_macro_mae'] <
        fold['baseline_extra_trees_history']['cell_macro_mae'] for fold in folds)
    return {'status': 'development-only, pre-specified weighting ablation',
            'difference_from_baseline': 'same 120-second history, imputer, ExtraTrees and '
                                        'outer splits; equal total training weight per strategy',
            'warning': 'the selected model remains unchanged; no independent test',
            'summary': summary, 'folds': folds}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'group_cv.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(evaluate(), indent=2), encoding='utf-8')
    print('WROTE', target)


if __name__ == '__main__':
    main()
