"""Causal learned-degradation-rate route for future SOH and RUL diagnostics.

Predicts a 50-cycle SOH change from signal history and out-of-fold SOH, then
projects that rate under a constant future operating strategy. This assumption
is explicit; it is not a simulation of a changed charging policy. The route
must beat the direct RUL model on held-out groups before deployment.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from train_rul_v2 import assisted_features, assess, eligible, oof_soh
from train_soh_v2 import fit as fit_soh, predict as predict_soh, outer_splits


OUT = Path('battery_models_v2/trajectory')
SOH_MODEL = 'extra_trees_history'
TRAIN_HORIZON = 50
MAX_RUL = 2500


def threshold_soh(cell):
    if cell.dataset in ('NASA', 'CALCE'):
        return .8
    if cell.dataset == 'MIT':
        return .88 / cell.reference_capacity_ah
    if cell.dataset == 'XJTU':
        return 1.6 / cell.reference_capacity_ah
    raise ValueError(cell.dataset)


def future_pairs(cell, horizon):
    cycle = cell.cycle
    current, future = [], []
    for i in range(max(15, cell.reference_count), len(cycle)):
        j = int(np.searchsorted(cycle, cycle[i]+horizon))
        if j >= len(cycle) or cycle[j]-cycle[i] > 1.5*horizon:
            continue
        if cell.endpoint_cycle is not None and cycle[j] > cell.endpoint_cycle:
            continue
        if not (cell.soh_available[i] and cell.soh_available[j]):
            continue
        current.append(i)
        future.append(j)
    return np.asarray(current, dtype=int), np.asarray(future, dtype=int)


def fit_rate(cells, ids, soh_predictions):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        current, future = future_pairs(cell, TRAIN_HORIZON)
        current, future = current[::5], future[::5]
        if not len(current):
            continue
        dt = cell.cycle[future]-cell.cycle[current]
        rate = (cell.soh[current]-cell.soh[future]) / dt
        x_parts.append(assisted_features(cell, soh_predictions[key])[current])
        y_parts.append(rate)
        flags = cell.capacity_review_flag[current] | cell.capacity_review_flag[future]
        weight = np.where(flags, .25, 1.).astype('float64')
        w_parts.append(weight / weight.sum() / len(ids))
    x, y, weights = map(np.concatenate, (x_parts, y_parts, w_parts))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=8,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weights)
    return {'imputer': imputer, 'model': model, 'train_horizon': TRAIN_HORIZON,
            'training_cell_ids': list(ids), 'data_contract_version': 2}


def predict_rate(bundle, cell, soh_prediction):
    x = assisted_features(cell, soh_prediction)
    return bundle['model'].predict(bundle['imputer'].transform(x))


def predict_trajectory(cell, soh_prediction, rate, start_index, horizons):
    """Future SOH under an unchanged rate/operating strategy scenario."""
    current = float(soh_prediction[start_index])
    return np.asarray([current-float(rate[start_index])*h for h in horizons], dtype='float32')


def predict_rul(cell, soh_prediction, rate):
    speed = np.asarray(rate)
    margin = np.asarray(soh_prediction)-threshold_soh(cell)
    finite_crossing = speed > 0
    estimate = np.where(finite_crossing, np.maximum(0, margin) /
                        np.maximum(speed, 1e-8), MAX_RUL)
    return np.clip(estimate, 0, MAX_RUL)


def forecast_score(cells, val_ids, soh_predictions, rate_predictions, horizon):
    rows = []
    for key in val_ids:
        cell = cells[key]
        current, future = future_pairs(cell, horizon)
        if not len(current):
            continue
        dt = cell.cycle[future]-cell.cycle[current]
        predicted = soh_predictions[key][current]-rate_predictions[key][current]*dt
        flat = soh_predictions[key][current]
        true = cell.soh[future]
        rows.append({'cell_id': key, 'n': int(len(current)),
                     'forecast_mae': float(np.mean(np.abs(predicted-true))),
                     'flat_soh_baseline_mae': float(np.mean(np.abs(flat-true)))})
    return {'cells': len(rows), 'cell_macro_mae': float(np.mean(
                [row['forecast_mae'] for row in rows])) if rows else None,
            'flat_soh_baseline_macro_mae': float(np.mean(
                [row['flat_soh_baseline_mae'] for row in rows])) if rows else None,
            'by_cell': rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cells = load_all()
    records = manifest['cells']
    summary = {'data_contract_version': 2, 'assumption': 'future strategy and learned local rate stay unchanged',
               'rate_target': 'next 50 source cycles SOH change per cycle', 'domains': {}}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        folder = OUT / domain.lower()
        folder.mkdir(parents=True, exist_ok=True)
        folds = []
        oof_rul = {}
        oof_splits = {}
        for name, train, val in outer_splits(records, domain):
            oof = oof_soh(cells, records, train)
            rate_model = fit_rate(cells, train, oof)
            soh_model = fit_soh(cells, train, SOH_MODEL)
            soh = {key: predict_soh(soh_model, cells[key]) for key in val}
            rate = {key: predict_rate(rate_model, cells[key], soh[key]) for key in val}
            observed_val = [key for key in val if cells[key].endpoint_cycle is not None]
            base_life = float(np.median([cells[key].endpoint_cycle for key in train
                                         if cells[key].endpoint_cycle is not None]))
            rul_predictions = {key: predict_rul(cells[key], soh[key], rate[key])
                               for key in observed_val}
            oof_rul.update(rul_predictions)
            oof_splits.update({key: name for key in observed_val})
            rul = assess(cells, observed_val, rul_predictions, base_life)
            fold = {'held_out': name, 'train_cells': len(train), 'validation_cells': len(val),
                    'forecast_50': forecast_score(cells, val, soh, rate, 50),
                    'forecast_100': forecast_score(cells, val, soh, rate, 100),
                    'rul': rul,
                    'unresolved_nonpositive_rate_fraction': float(np.mean(np.concatenate(
                        [rate[key][eligible(cells[key])] <= 0 for key in observed_val])))}
            folds.append(fold)
            future_mae = fold['forecast_50']['cell_macro_mae']
            print(domain, name, 'SOH+50', round(future_mae, 4) if future_mae is not None else None,
                  'RUL', round(rul['cell_macro_mae_cycles'], 1), flush=True)
        (folder / 'group_cv.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
        ids = [row['cell_id'] for row in records if row['dataset'] == domain]
        target, estimate, split, cell_id, cycle, endpoint = [], [], [], [], [], []
        for key in ids:
            cell = cells[key]
            if cell.endpoint_cycle is None:
                continue
            keep = eligible(cell)
            n = int(keep.sum())
            target.append(cell.rul_cycles[keep])
            estimate.append(oof_rul[key][keep])
            split.append(np.full(n, oof_splits[key]))
            cell_id.append(np.full(n, key))
            cycle.append(cell.cycle[keep])
            endpoint.append(np.full(n, cell.endpoint_cycle))
        np.savez_compressed(folder / 'development_oof_rul.npz',
                            target=np.concatenate(target), estimate=np.concatenate(estimate),
                            held_out=np.concatenate(split), cell_id=np.concatenate(cell_id),
                            cycle=np.concatenate(cycle), endpoint=np.concatenate(endpoint))
        oof = oof_soh(cells, records, ids)
        model = fit_rate(cells, ids, oof)
        joblib.dump(model, folder / 'rate_model.joblib')
        metrics = {'forecast_50_macro_mae': float(np.mean(
                       [fold['forecast_50']['cell_macro_mae'] for fold in folds
                        if fold['forecast_50']['cell_macro_mae'] is not None])),
                   'forecast_100_macro_mae': float(np.mean(
                       [fold['forecast_100']['cell_macro_mae'] for fold in folds
                        if fold['forecast_100']['cell_macro_mae'] is not None])),
                   'rul_macro_mae_cycles': float(np.mean(
                       [fold['rul']['cell_macro_mae_cycles'] for fold in folds])),
                   'training_cells': len(ids),
                   'endpoint_threshold': {'NASA': '0.8 * reference Ah',
                                          'CALCE': '0.8 * reference Ah',
                                          'MIT': '0.88 Ah', 'XJTU': '1.6 Ah nominal'}[domain],
                   'selection': 'diagnostic route only; direct model remains selected unless consistently better'}
        summary['domains'][domain] = metrics
        print('SUMMARY', domain, metrics, flush=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
