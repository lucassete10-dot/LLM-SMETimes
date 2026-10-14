"""Development test of an empirical, end-to-end capacity-curve regressor.

Inspired by Li et al. (Applied Energy 2025), but deliberately a small linear
adaptation, not a reproduction of their reported experiments. A train-cell
prefix predicts nonnegative linear/cubic fade coefficients. The coefficients
are fitted against all future SOH observations with equal cell weights.
Validation holds out a complete XJTU operating group. No future features or
true validation capacity enter inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_all
from train_rul_v2 import signal_features
from train_soh_v2 import fit as fit_soh, outer_splits, predict as predict_soh
from train_trajectory_v2 import threshold_soh


OUT = Path('battery_models_v2/empirical_curve_challenger')
MAX_HORIZON = 2500


def first_origin(cell):
    # Availability masks are labels, and are unknown at online inference.
    origin = max(15, cell.reference_count)
    return origin if origin < len(cell.cycle) else None


def early_features(cell, origin, feature_mode):
    if feature_mode == 'full':
        return signal_features(cell)[origin]
    if feature_mode == 'relative':
        age = float(cell.cycle[origin])
        return np.r_[cell.delta[origin], cell.history[origin], age,
                     np.log1p(age)].astype('float32')
    raise ValueError(f'unknown feature mode: {feature_mode}')


def training_pairs(cells, ids, feature_mode='full'):
    """Return one feature row per cell and balanced future delta targets."""
    features, pairs, used = [], [], []
    for key in ids:
        cell = cells[key]
        origin = first_origin(cell)
        if (origin is None or cell.endpoint_cycle is None or
                not cell.soh_available[origin]):
            continue
        future = np.flatnonzero((cell.cycle > cell.cycle[origin]) &
                                (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
        if len(future) < 2:
            continue
        # Subsample only training labels, retaining the beginning and endpoint.
        future = future[np.unique(np.linspace(0, len(future)-1,
                                            min(80, len(future)), dtype=int))]
        horizon = cell.cycle[future]-cell.cycle[origin]
        delta = cell.soh[origin]-cell.soh[future]
        weight = np.where(cell.capacity_review_flag[future], .25, 1.).astype(float)
        features.append(early_features(cell, origin, feature_mode))
        endpoint_horizon = int(cell.endpoint_cycle-cell.cycle[origin])
        endpoint_delta = float(cell.soh[origin]-threshold_soh(cell))
        pairs.append((horizon, delta, weight/weight.sum(),
                      endpoint_horizon, endpoint_delta))
        used.append(key)
    if len(used) < 3:
        raise ValueError('at least three observed training cells required')
    return np.asarray(features), pairs, used


def design(z, horizon, scale):
    """Linear map from early features to two capacity-fade coefficients."""
    u = np.asarray(horizon, dtype=float)/scale
    v = np.r_[1., np.asarray(z, dtype=float)]
    return np.column_stack((u[:, None]*v, (u**3)[:, None]*v))


def fit_empirical(cells, ids, alpha=10.0, endpoint_weight=0.0,
                  feature_mode='full'):
    features, pairs, used = training_pairs(cells, ids, feature_mode)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(features)
    imputed = imputer.transform(features)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(imputed)
    scaled = np.clip(scaler.transform(imputed), -6, 6)
    n_components = min(8, len(used)-2, scaled.shape[1])
    pca = PCA(n_components=n_components).fit(scaled)
    z = pca.transform(scaled)
    # Life scale is determined strictly from outer-training cells.
    scale = float(np.median([cell.endpoint_cycle for cell in
                             (cells[key] for key in used)]))
    x, y, weights = [], [], []
    for row, (horizon, delta, weight, endpoint_horizon,
              endpoint_delta) in zip(z, pairs):
        x.append(design(row, horizon, scale))
        y.append(delta)
        weights.append(weight/len(used))
        if endpoint_weight:
            x.append(design(row, [endpoint_horizon], scale))
            y.append(np.array([endpoint_delta]))
            weights.append(np.array([endpoint_weight/len(used)]))
    x, y, weights = map(np.concatenate, (x, y, weights))
    model = Ridge(alpha=alpha, fit_intercept=False).fit(
        x, y, sample_weight=weights/weights.sum()*len(y))
    return {'imputer': imputer, 'scaler': scaler, 'pca': pca, 'scale': scale,
            'model': model, 'training_cells': used, 'training_rows': len(y),
            'alpha': alpha, 'endpoint_weight': endpoint_weight,
            'feature_mode': feature_mode}


def predict_curve(bundle, cell, current_soh, origin, max_horizon=MAX_HORIZON):
    features = early_features(cell, origin, bundle['feature_mode'])[None, :]
    scaled = np.clip(bundle['scaler'].transform(
        bundle['imputer'].transform(features)), -6, 6)
    z = bundle['pca'].transform(scaled)[0]
    n = len(z)+1
    coefficients = np.maximum(0, bundle['model'].coef_.reshape(2, n) @
                              np.r_[1., z])
    u = np.arange(1, max_horizon+1, dtype=float)/bundle['scale']
    curve = np.clip(float(current_soh)-coefficients[0]*u-
                    coefficients[1]*u**3, 0, float(current_soh))
    return curve, coefficients


def score_cell(cell, curve, origin, coefficients):
    future = np.flatnonzero((cell.cycle > cell.cycle[origin]) &
                            (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
    if len(future) < 2:
        return None
    horizon = cell.cycle[future]-cell.cycle[origin]
    if horizon.max() > len(curve):
        raise ValueError(f'{cell.cell_id}: increase MAX_HORIZON')
    actual = cell.soh[future]
    crossing = np.flatnonzero(curve <= threshold_soh(cell))
    # Non-crossing predictions are failures, not silently excluded from MAPE.
    predicted_eol = (int(cell.cycle[origin]+crossing[0]+1)
                     if len(crossing) else int(cell.cycle[origin]+len(curve)))
    return {'cell_id': cell.cell_id, 'group': cell.group,
            'origin_cycle': int(cell.cycle[origin]),
            'curve_mape_percent': float(100*np.mean(np.abs(
                curve[horizon-1]-actual)/actual)),
            'eol_ape_percent': float(100*abs(predicted_eol-cell.endpoint_cycle)/
                                     cell.endpoint_cycle),
            'crossed_within_grid': bool(len(crossing)),
            'true_endpoint_cycle': int(cell.endpoint_cycle),
            'predicted_endpoint_cycle': predicted_eol,
            'fade_coefficients': coefficients.tolist()}


def evaluate_domain(domain, manifest, cells, endpoint_weight=0.0,
                    feature_mode='full'):
    rows = [row for row in manifest['cells'] if row['dataset'] == domain]
    folds = []
    for group, train, val in outer_splits(rows, domain):
        empirical = fit_empirical(cells, train, endpoint_weight=endpoint_weight,
                                  feature_mode=feature_mode)
        soh_model = fit_soh(cells, train, 'extra_trees_history')
        scored = []
        for key in val:
            cell = cells[key]
            origin = first_origin(cell)
            if origin is None or cell.endpoint_cycle is None:
                continue
            current = predict_soh(soh_model, cell)[origin]
            curve, coefficients = predict_curve(empirical, cell, current, origin)
            result = score_cell(cell, curve, origin, coefficients)
            if result is not None:
                scored.append(result)
        folds.append({'held_out': group, 'training_cells': len(empirical['training_cells']),
                      'scored_cells': len(scored), 'by_cell': scored})
        print(domain, group, 'cells', len(scored), 'curve MAPE',
              round(float(np.mean([r['curve_mape_percent'] for r in scored])), 2)
              if scored else None, 'EOL MAPE',
              round(float(np.mean([r['eol_ape_percent'] for r in scored])), 2)
              if scored else None, flush=True)
    scored = [row for fold in folds for row in fold['by_cell']]
    return {'summary': {'scored_cells': len(scored),
                        'curve_mape_percent': float(np.mean([
                            row['curve_mape_percent'] for row in scored])),
                        'eol_mape_percent': float(np.mean([
                            row['eol_ape_percent'] for row in scored])),
                        'crossing_coverage': float(np.mean([
                            row['crossed_within_grid'] for row in scored]))},
            'folds': folds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=('XJTU', 'MIT'),
                        default=['XJTU'])
    parser.add_argument('--endpoint-weight', type=float, choices=(0.0, 0.5),
                        default=0.0)
    parser.add_argument('--feature-mode', choices=('full', 'relative'), default='full')
    args = parser.parse_args()
    manifest, cells = load_all()
    report = {'status': 'unpromoted development challenger; not independent test',
              'method': 'Li-inspired end-to-end linear/cubic empirical capacity curve',
              'input': 'capacity-free signal prefix plus outer-training SOH model',
              'future_assumption': 'observed operating regime continues',
              'endpoint_anchor_weight': args.endpoint_weight,
              'feature_mode': args.feature_mode,
              'endpoint_warning': 'MIT and XJTU endpoints are source-defined, not exact per-cell 80%',
              'domains': {}}
    for domain in args.domains:
        report['domains'][domain] = evaluate_domain(
            domain, manifest, cells, endpoint_weight=args.endpoint_weight,
            feature_mode=args.feature_mode)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = ('_endpoint_anchor' if args.endpoint_weight else '') + (
        '_relative' if args.feature_mode == 'relative' else '')
    path = OUT / ('development_cv_'+'-'.join(d.lower() for d in args.domains)+suffix+'.json')
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', path)


if __name__ == '__main__':
    main()
