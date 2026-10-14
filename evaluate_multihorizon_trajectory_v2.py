"""Development-only multi-horizon SOH forecast under unchanged future strategy.

The model predicts an average SOH decline rate over a requested future cycle
span. Training SOH inputs are cross-fitted inside each outer training fold;
validation features and labels stay on held-out batches/strategies.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from train_rul_v2 import assisted_features, oof_soh
from train_soh_v2 import fit as fit_soh, outer_splits, predict as predict_soh
from train_trajectory_v2 import future_pairs


OUT = Path('battery_models_v2/multihorizon_trajectory_diagnostic.json')
HORIZONS = (50, 100, 200, 400)


def append_horizon(x, dt):
    dt = np.broadcast_to(np.asarray(dt, dtype='float32'), (len(x),))
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError('forecast horizon must be finite and positive')
    return np.column_stack((x, dt, np.log1p(dt))).astype('float32')


def training_parts(cell, soh_estimate, horizons=HORIZONS):
    base = assisted_features(cell, soh_estimate)
    parts = []
    for horizon in horizons:
        current, future = future_pairs(cell, horizon)
        current, future = current[::5], future[::5]
        if not len(current):
            continue
        dt = (cell.cycle[future] - cell.cycle[current]).astype('float32')
        rate = (cell.soh[current] - cell.soh[future]) / dt
        flagged = cell.capacity_review_flag[current] | cell.capacity_review_flag[future]
        weight = np.where(flagged, .25, 1.).astype('float64')
        parts.append((append_horizon(base[current], dt), rate,
                      weight / weight.sum()))
    return [(x, y, w / len(parts)) for x, y, w in parts]


def fit_multihorizon(cells, ids, soh_estimates):
    collected = [[], [], []]
    eligible_cells = 0
    for key in ids:
        parts = training_parts(cells[key], soh_estimates[key])
        if not parts:
            continue
        eligible_cells += 1
        for x, y, weight in parts:
            for target, value in zip(collected, (x, y, weight)):
                target.append(value)
    if not eligible_cells:
        raise ValueError('no future SOH pairs in training fold')
    x, y, weight = [np.concatenate(item) for item in collected]
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=8,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weight / eligible_cells)
    return {'imputer': imputer, 'model': model, 'training_cells': eligible_cells,
            'training_rows': len(y)}


def forecast_cell(bundle, cell, soh_estimate, horizon):
    current, future = future_pairs(cell, horizon)
    if not len(current):
        return None
    dt = cell.cycle[future] - cell.cycle[current]
    x = append_horizon(assisted_features(cell, soh_estimate)[current], dt)
    rate = bundle['model'].predict(bundle['imputer'].transform(x))
    prediction = soh_estimate[current] - rate * dt
    actual = cell.soh[future]
    return {'n': int(len(current)),
            'forecast_mae': float(np.mean(np.abs(prediction-actual))),
            'flat_soh_mae': float(np.mean(np.abs(soh_estimate[current]-actual))),
            'signed_bias': float(np.mean(prediction-actual))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=('MIT', 'XJTU'),
                        default=['MIT', 'XJTU'])
    args = parser.parse_args()
    manifest, cells = load_all()
    report = {'status': 'development diagnostic; not selected or deployed',
              'future_scenario': 'same operating strategy as observed history',
              'train_horizons_cycles': HORIZONS, 'subsample_stride': 5,
              'domains': {}}
    for domain in args.domains:
        folds = []
        for group, train, validation in outer_splits(manifest['cells'], domain):
            oof = oof_soh(cells, manifest['cells'], train)
            trajectory = fit_multihorizon(cells, train, oof)
            soh_model = fit_soh(cells, train, 'extra_trees_history')
            scores = {}
            for horizon in HORIZONS:
                by_cell = {}
                for key in validation:
                    soh = predict_soh(soh_model, cells[key])
                    result = forecast_cell(trajectory, cells[key], soh, horizon)
                    if result is not None:
                        by_cell[key] = result
                scores[str(horizon)] = {
                    'cells': len(by_cell),
                    'cell_macro_mae': float(np.mean([v['forecast_mae'] for v in by_cell.values()]))
                    if by_cell else None,
                    'flat_soh_macro_mae': float(np.mean([v['flat_soh_mae'] for v in by_cell.values()]))
                    if by_cell else None,
                    'cell_macro_signed_bias': float(np.mean([v['signed_bias'] for v in by_cell.values()]))
                    if by_cell else None,
                    'by_cell': by_cell}
            folds.append({'held_out': group, 'train_cells_with_future_pairs':
                          trajectory['training_cells'], 'training_rows':
                          trajectory['training_rows'], 'horizons': scores})
            print(domain, group, {h: round(scores[str(h)]['cell_macro_mae'], 4)
                                  if scores[str(h)]['cell_macro_mae'] is not None else None
                                  for h in HORIZONS}, flush=True)
        summary = {}
        for horizon in HORIZONS:
            rows = [row['horizons'][str(horizon)] for row in folds]
            nonempty = [row for row in rows if row['cells']]
            summary[str(horizon)] = {
                'folds_with_targets': len(nonempty),
                'equal_fold_macro_mae': float(np.mean([row['cell_macro_mae'] for row in nonempty]))
                if nonempty else None,
                'equal_cell_macro_mae': float(np.mean([cell['forecast_mae']
                    for row in rows for cell in row['by_cell'].values()])) if nonempty else None,
                'flat_soh_equal_cell_macro_mae': float(np.mean([cell['flat_soh_mae']
                    for row in rows for cell in row['by_cell'].values()])) if nonempty else None,
            }
        report['domains'][domain] = {'summary': summary, 'folds': folds}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
