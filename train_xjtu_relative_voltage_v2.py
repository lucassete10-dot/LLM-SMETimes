"""XJTU SOH ablation: add charge-prefix relative-voltage crossing features.

The 0.05–0.10 V rise window is calculated from five already-audited samples
within the first 120 seconds of each charge stage. Missing crossings remain
missing (plus an availability bit); only training cells fit the imputer. This
ablation does not modify the frozen selected SOH model.
"""

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from train_soh_v2 import outer_splits, score


OUT = Path('battery_models_v2/xjtu_relative_voltage')
TIMES = np.arange(0, 121, 30, dtype='float32')


def first_crossing(voltage, rise_target):
    rise = np.maximum.accumulate(voltage-voltage[0])
    if not np.isfinite(rise).all() or rise[-1] < rise_target:
        return np.nan
    index = int(np.searchsorted(rise, rise_target))
    if index == 0:
        return float(TIMES[0])
    low, high = float(rise[index-1]), float(rise[index])
    if high == low:
        return float(TIMES[index])
    return float(TIMES[index-1] + (rise_target-low)/(high-low)*30)


def relative_features(cell):
    voltage = cell.raw[:, :5]
    rows = []
    for sample in voltage:
        low = first_crossing(sample, .05)
        high = first_crossing(sample, .10)
        available = float(np.isfinite(low) and np.isfinite(high))
        duration = high-low if available else np.nan
        slope = .05/duration if available and duration > 0 else np.nan
        rows.append([low, high, duration, slope, available])
    x = np.asarray(rows, dtype='float32')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        anchor = np.nanmedian(x[:cell.reference_count], axis=0)
        delta = x-anchor
        history = np.stack([np.nanmean(delta[max(0, i-4):i+1], axis=0)
                            for i in range(len(x))])
    return np.column_stack((cell.matrix('signal_history'), x, delta, history)).astype('float32')


def fit(cells, features, ids):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        x_parts.append(features[key][keep])
        y_parts.append(cell.soh[keep])
        w = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        w_parts.append(w/w.sum()/len(ids))
    x, y, weight = map(np.concatenate, (x_parts, y_parts, w_parts))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weight)
    return {'imputer': imputer, 'model': model, 'feature_dimension': x.shape[1],
            'training_cell_ids': list(ids), 'data_contract_version': 2}


def predict(bundle, features):
    return bundle['model'].predict(bundle['imputer'].transform(features))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'XJTU'}
    features = {key: relative_features(cell) for key, cell in cells.items()}
    folds = []
    for group, train, val in outer_splits(manifest['cells'], 'XJTU'):
        model = fit(cells, features, train)
        metric = score(cells, {key: predict(model, features[key]) for key in val}, val)
        folds.append({'held_out': group, 'score': metric})
        print(group, round(metric['cell_macro_mae'], 5),
              'signed_bias', round(metric['cell_macro_signed_bias'], 5), flush=True)
    (OUT / 'group_cv.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
    ids = list(cells)
    joblib.dump(fit(cells, features, ids), OUT / 'extra_trees_relative_voltage.joblib')
    baseline = json.loads(Path('battery_models_v2/soh/xjtu/training_manifest.json').read_text())
    result = {'data_contract_version': 2,
              'feature': 'first-120-s charge time to +0.05 and +0.10 V versus initial voltage',
              'group_cv_macro_mae': float(np.mean([fold['score']['cell_macro_mae'] for fold in folds])),
              'frozen_baseline_group_cv_macro_mae': baseline['group_cv_macro_mae']['extra_trees_history'],
              'held_out_groups': len(folds), 'training_cells': len(ids),
              'selection': 'diagnostic only; frozen model is not changed'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('SUMMARY', result, flush=True)


if __name__ == '__main__':
    main()
