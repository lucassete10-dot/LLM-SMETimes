"""Joint future-SOH-curve and EOL prediction from causal multihorizon signals.

The future curve is predicted at all source cycles and made bounded and
nonincreasing. Its declared-threshold crossing is the RUL prediction; there
is no separate endpoint model to contradict the curve. Every outer split
holds out a whole batch/strategy. This is development work, not external test.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from evaluate_multihorizon_trajectory_v2 import append_horizon, training_parts
from train_rul_v2 import assisted_features, eligible, oof_soh
from train_soh_v2 import fit as fit_soh, outer_splits, predict as predict_soh
from train_trajectory_v2 import threshold_soh


OUT = Path('battery_models_v2/joint_future_curve_challenger')
HORIZONS = (25, 50, 100, 200, 400, 800, 1200)
MAX_FORECAST_CYCLES = 2500


def fit_curve(cells, ids, soh_oof):
    x_parts, y_parts, w_parts = [], [], []
    active = 0
    for key in ids:
        cell = cells[key]
        # Reuse the existing leakage-controlled pair builder, but include
        # longer horizons and reduce correlated within-cell sampling.
        parts = training_parts(cell, soh_oof[key], horizons=HORIZONS)
        if not parts:
            continue
        active += 1
        for x, y, weight in parts:
            x_parts.append(x[::2])
            y_parts.append(y[::2])
            w_parts.append(weight[::2])
    if active < 2:
        raise ValueError('at least two training cells with future pairs required')
    x, y, weight = map(np.concatenate, (x_parts, y_parts, w_parts))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=80, min_samples_leaf=10,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weight/active)
    return {'imputer': imputer, 'model': model, 'training_cell_ids': list(ids),
            'training_rows': len(y), 'horizons': HORIZONS, 'data_contract_version': 2}


def bounded_curve(current_soh, rate_by_horizon, horizon):
    """A physically bounded, nonincreasing SOH trajectory from rate forecasts."""
    horizon = np.asarray(horizon, dtype='float64')
    rates = np.asarray(rate_by_horizon, dtype='float64')
    if horizon.ndim != 1 or len(horizon) != len(rates) or np.any(np.diff(horizon) <= 0):
        raise ValueError('increasing one-dimensional horizons required')
    if not np.isfinite(current_soh) or not np.isfinite(rates).all():
        raise ValueError('non-finite SOH or degradation rate')
    raw = float(current_soh)-rates*horizon
    return np.minimum.accumulate(np.clip(raw, 0, float(current_soh)))


def predict_curve(bundle, cell, soh_estimate, origin_index, max_horizon=MAX_FORECAST_CYCLES):
    horizon = np.arange(1, max_horizon+1, dtype='float32')
    base = assisted_features(cell, soh_estimate)[origin_index:origin_index+1]
    x = append_horizon(np.repeat(base, len(horizon), axis=0), horizon)
    rate = bundle['model'].predict(bundle['imputer'].transform(x))
    return bounded_curve(soh_estimate[origin_index], rate, horizon)


def evaluate_cell(cell, estimate, curve):
    usable = np.flatnonzero(eligible(cell))
    if not len(usable):
        return None
    origin_index = int(usable[0])
    origin_cycle = int(cell.cycle[origin_index])
    future = np.flatnonzero((cell.cycle > origin_cycle) &
                            (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
    if len(future) < 2:
        return None
    horizon = cell.cycle[future]-origin_cycle
    if horizon.max() > len(curve):
        raise ValueError(f'{cell.cell_id}: forecast grid too short')
    predicted = curve[horizon-1]
    truth = cell.soh[future]
    crossing = np.flatnonzero(curve <= threshold_soh(cell))
    eol_cycle = origin_cycle + int(crossing[0])+1 if len(crossing) else None
    return {'cell_id': cell.cell_id, 'group': cell.group,
            'origin_cycle': origin_cycle, 'future_points': len(future),
            'curve_mape_percent': float(100*np.mean(np.abs(predicted-truth)/truth)),
            'curve_mae_soh': float(np.mean(np.abs(predicted-truth))),
            'physical_violations': int(np.count_nonzero(np.diff(curve) > 1e-9) +
                                       np.count_nonzero(curve < 0) +
                                       np.count_nonzero(curve > estimate[origin_index]+1e-9)),
            'true_endpoint_cycle': cell.endpoint_cycle,
            'predicted_endpoint_cycle': eol_cycle,
            'eol_ape_percent': (float(100*abs(eol_cycle-cell.endpoint_cycle)/cell.endpoint_cycle)
                                if eol_cycle is not None else None)}


def evaluate_domain(domain, manifest, all_cells, save_full_model=False):
    records = [record for record in manifest['cells'] if record['dataset'] == domain]
    cells = {record['cell_id']: all_cells[record['cell_id']] for record in records}
    folds = []
    for group, train, val in outer_splits(records, domain):
        inner_soh = oof_soh(cells, records, train)
        curve_model = fit_curve(cells, train, inner_soh)
        soh_model = fit_soh(cells, train, 'extra_trees_history')
        rows = []
        for key in val:
            cell = cells[key]
            if cell.endpoint_cycle is None:
                continue
            usable = np.flatnonzero(eligible(cell))
            if not len(usable):
                continue
            soh = predict_soh(soh_model, cell)
            curve = predict_curve(curve_model, cell, soh, int(usable[0]))
            result = evaluate_cell(cell, soh, curve)
            if result is not None:
                rows.append(result)
        fold = {'held_out': group, 'train_cells': len(train),
                'validation_endpoint_cells': sum(cells[key].endpoint_cycle is not None for key in val),
                'scored_cells': len(rows), 'training_rows': curve_model['training_rows'],
                'curve_mape_percent': float(np.mean([r['curve_mape_percent'] for r in rows]))
                if rows else None,
                'eol_scored_cells': sum(r['eol_ape_percent'] is not None for r in rows),
                'eol_mape_percent': float(np.mean([r['eol_ape_percent'] for r in rows
                                                  if r['eol_ape_percent'] is not None]))
                if any(r['eol_ape_percent'] is not None for r in rows) else None,
                'by_cell': rows}
        folds.append(fold)
        print(domain, group, 'curve', round(fold['curve_mape_percent'], 2)
              if rows else None, 'EOL', round(fold['eol_mape_percent'], 2)
              if fold['eol_mape_percent'] is not None else None,
              'EOL coverage', fold['eol_scored_cells'], '/', len(rows), flush=True)
    rows = [r for fold in folds for r in fold['by_cell']]
    eol = [r['eol_ape_percent'] for r in rows if r['eol_ape_percent'] is not None]
    summary = {'scored_cells': len(rows),
               'endpoint_cells': sum(c.endpoint_cycle is not None for c in cells.values()),
               'equal_cell_curve_mape_percent': float(np.mean([
                   r['curve_mape_percent'] for r in rows])),
               'equal_fold_curve_mape_percent': float(np.mean([
                   fold['curve_mape_percent'] for fold in folds if fold['scored_cells']])),
               'worst_cell_curve_mape_percent': float(max(
                   r['curve_mape_percent'] for r in rows)),
               'eol_scored_cells': len(eol),
               'eol_mape_percent': float(np.mean(eol)) if eol else None,
               'physical_violation_cells': sum(r['physical_violations'] > 0 for r in rows)}
    if save_full_model:
        inner_soh = oof_soh(cells, records, list(cells))
        fitted = fit_curve(cells, list(cells), inner_soh)
        OUT.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, OUT / f'{domain.lower()}_joint_candidate.joblib')
    return {'summary': summary, 'folds': folds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=('MIT', 'XJTU'), default=['XJTU'])
    parser.add_argument('--save-models', action='store_true')
    args = parser.parse_args()
    manifest, cells = load_all()
    report = {'status': 'development whole-batch/strategy CV; not independent test',
              'future_assumption': 'observed operating regime continues',
              'input': 'capacity-free current/past signal and out-of-fold SOH estimate',
              'method': 'causal multihorizon rate regressor; bounded monotone curve; '
                        'RUL from first declared-threshold crossing',
              'train_horizons': HORIZONS, 'domains': {}}
    for domain in args.domains:
        report['domains'][domain] = evaluate_domain(domain, manifest, cells, args.save_models)
    OUT.mkdir(parents=True, exist_ok=True)
    name = '-'.join(domain.lower() for domain in args.domains)
    path = OUT / f'development_cv_{name}.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', path)


if __name__ == '__main__':
    main()
