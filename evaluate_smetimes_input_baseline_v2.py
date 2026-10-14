"""XJTU SOH ablation of the 10x9 SMETimes input without an LLM backbone.

Uses the same six whole-strategy folds, label masks, cell-balanced weights,
and ExtraTrees settings as the selected 120s SOH model. This distinguishes
input-resampling loss from learned-representation effects.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_smetimes_signal_v2 import build_sequence
from battery_v2_features import derive_signal_features, load_all
from train_soh_v2 import outer_splits, score


OUT = Path('battery_models_v2/original_smetimes_xjtu/input_only_cv.json')


def input_features(cell):
    sequence, context = build_sequence('XJTU', cell.raw)
    raw = np.column_stack((sequence.reshape(len(sequence), -1), context)).astype('float32')
    delta, history = derive_signal_features(raw, cell.reference_count)
    return np.column_stack((raw, delta, history)).astype('float32')


def fit(cells, features, ids):
    inputs, labels, weights = [], [], []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        inputs.append(features[key][keep])
        labels.append(cell.soh[keep])
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        weights.append(weight/weight.sum()/len(ids))
    x, y, w = map(np.concatenate, (inputs, labels, weights))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=w)
    return imputer, model


def evaluate():
    manifest, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'XJTU'}
    features = {key: input_features(cell) for key, cell in cells.items()}
    stored = {fold['held_out']: fold['extra_trees_history'] for fold in json.loads(
        Path('battery_models_v2/soh/xjtu/group_cv.json').read_text(encoding='utf-8'))}
    folds = []
    for group, train, val in outer_splits(manifest['cells'], 'XJTU'):
        imputer, model = fit(cells, features, train)
        predictions = {key: model.predict(imputer.transform(features[key])) for key in val}
        metric = score(cells, predictions, val)
        folds.append({'held_out': group, 'baseline_120s': stored[group],
                      'flat_smetimes_input': metric})
        print(group, 'flat', round(metric['cell_macro_mae'], 5),
              'baseline', round(stored[group]['cell_macro_mae'], 5), flush=True)
    summary = {name: {
        'equal_fold_macro_mae': float(np.mean([row[name]['cell_macro_mae'] for row in folds])),
        'worst_cell_mae': float(max(cell['mae'] for row in folds
                                    for cell in row[name]['by_cell'])),
        'folds_better_than_baseline': sum(row[name]['cell_macro_mae'] <
            row['baseline_120s']['cell_macro_mae'] for row in folds),
    } for name in ('baseline_120s', 'flat_smetimes_input')}
    return {'status': 'development-only input ablation; no model deployed',
            'input': 'same 10x9 physical SMETimes adapter plus four static context values',
            'estimator': 'same ExtraTrees parameters and cell weights as selected SOH model',
            'summary': summary, 'folds': folds}


if __name__ == '__main__':
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate()
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', OUT, flush=True)
