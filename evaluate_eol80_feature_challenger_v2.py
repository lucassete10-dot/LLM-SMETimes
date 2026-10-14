"""Development-only 80%-EOL challenger using causal, robust signal history.

This is inspired by the feature-denoising and early-life prediction papers,
but is not an implementation or validation of PBT. No target capacity, SOH,
or eventual lifetime is used as a model input. Existing frozen models are not
modified. Checkpoint coverage is explicit because XJTU diagnostics are sparse.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from train_rul_v2 import assess, eligible, signal_features
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/eol80_feature_challenger')
CHECKPOINTS = (20, 50, 100)
METADATA = ('charge_rate_c', 'discharge_rate_c',
            'diagnostic_discharge_rate_c', 'ambient_temperature_c')


def causal_features(cell, record):
    """Use only current/past signal rows and known operating metadata."""
    raw = cell.raw.astype('float64')
    recent = np.empty_like(raw)
    change = np.empty_like(raw)
    variation = np.empty_like(raw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(len(raw)):
            window = raw[max(0, i - 4):i + 1]
            median = np.nanmedian(window, axis=0)
            older = np.nanmedian(raw[max(0, i - 9):max(1, i - 4)], axis=0)
            span = max(1, int(cell.cycle[i] - cell.cycle[max(0, i - 5)]))
            recent[i] = median
            change[i] = (median - older) / span
            variation[i] = np.nanmedian(np.abs(window - median), axis=0)
    values = np.array([np.nan if record.get(name) is None else float(record[name])
                       for name in METADATA], dtype='float64')
    metadata = np.repeat(values[None, :], len(raw), axis=0)
    missing = np.repeat(np.isnan(values)[None, :].astype('float64'), len(raw), axis=0)
    return np.column_stack((signal_features(cell), recent, change, variation,
                            metadata, missing)).astype('float32')


def fit(cells, records, ids, features, target='raw_eol'):
    if target not in ('raw_eol', 'log_rul'):
        raise ValueError(target)
    parts, targets, weights = [], [], []
    for key in ids:
        cell = cells[key]
        if cell.endpoint_cycle is None:
            continue
        rows = np.flatnonzero(eligible(cell))[::5]
        if not len(rows):
            continue
        # Equal total weight per cell prevents densely sampled cells dominating.
        weight = np.where(cell.capacity_review_flag[rows], .25, 1.).astype('float64')
        weight /= weight.sum()
        parts.append(features[key][rows])
        if target == 'raw_eol':
            targets.append(np.full(len(rows), cell.endpoint_cycle, dtype='float64'))
        else:
            targets.append(np.log1p(cell.rul_cycles[rows].astype('float64')))
        weights.append(weight)
    if not parts:
        raise ValueError('training fold has no observed endpoints')
    x = np.concatenate(parts)
    y = np.concatenate(targets)
    w = np.concatenate(weights)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5,
                                max_features=.7, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=w)
    return {'imputer': imputer, 'model': model, 'training_cell_ids': list(ids),
            'data_contract_version': 2, 'prediction_target': target}


def predict(bundle, cell, x):
    predicted = bundle['model'].predict(bundle['imputer'].transform(x))
    if bundle['prediction_target'] == 'log_rul':
        return np.maximum(0, np.expm1(predicted))
    endpoint = np.maximum(predicted, cell.cycle)
    return endpoint - cell.cycle


def checkpoint_score(cells, ids, predictions, checkpoint, baseline_life):
    rows = []
    for key in ids:
        cell = cells[key]
        if cell.endpoint_cycle is None or cell.endpoint_cycle <= checkpoint:
            continue
        available = np.flatnonzero(eligible(cell) & (cell.cycle <= checkpoint))
        if not len(available):
            continue
        i = int(available[-1])
        predicted_endpoint = cell.cycle[i] + predictions[key][i]
        error = predicted_endpoint - cell.endpoint_cycle
        rows.append({'cell_id': key, 'observation_cycle': int(cell.cycle[i]),
                     'true_endpoint_cycle': int(cell.endpoint_cycle),
                     'predicted_endpoint_cycle': float(predicted_endpoint),
                     'absolute_error_cycles': float(abs(error)),
                     'life_absolute_percentage_error': float(abs(error) / cell.endpoint_cycle),
                     'median_life_baseline_error_cycles': float(
                         abs(baseline_life - cell.endpoint_cycle))})
    if not rows:
        return {'cells': 0, 'mae_cycles': None, 'life_mape': None,
                'median_life_baseline_mae_cycles': None, 'by_cell': []}
    return {'cells': len(rows),
            'mae_cycles': float(np.mean([r['absolute_error_cycles'] for r in rows])),
            'life_mape': float(np.mean([r['life_absolute_percentage_error'] for r in rows])),
            'median_life_baseline_mae_cycles': float(np.mean(
                [r['median_life_baseline_error_cycles'] for r in rows])),
            'by_cell': rows}


def evaluate(domains, save_models=False, target='raw_eol'):
    manifest, cells = load_all()
    records = {r['cell_id']: r for r in manifest['cells']}
    features = {key: causal_features(cell, records[key]) for key, cell in cells.items()
                if cell.dataset in domains}
    report = {'status': 'development-only challenger; frozen models unchanged',
              'data_contract_version': 2,
              'target': 'existing contract observed 80%-EOL cycle; no censored endpoint imputation',
              'method': 'causal robust median, cycle-normalized change, MAD, known condition metadata; ExtraTrees regression',
              'prediction_target': target,
              'validation': 'whole-cell NASA/CALCE; whole-batch MIT; whole-strategy XJTU',
              'checkpoints_source_cycles': list(CHECKPOINTS), 'domains': {}}
    for domain in domains:
        folds = []
        for name, train_ids, val_ids in outer_splits(manifest['cells'], domain):
            observed_train = [key for key in train_ids if cells[key].endpoint_cycle is not None]
            observed_val = [key for key in val_ids if cells[key].endpoint_cycle is not None]
            baseline_life = float(np.median([cells[key].endpoint_cycle for key in observed_train]))
            bundle = fit(cells, records, observed_train, features, target)
            predictions = {key: predict(bundle, cells[key], features[key]) for key in observed_val}
            fold = {'held_out': name, 'train_cells': len(observed_train),
                    'validation_cells': len(observed_val),
                    'all_eligible': assess(cells, observed_val, predictions, baseline_life),
                    'checkpoints': {str(t): checkpoint_score(cells, observed_val,
                                                              predictions, t, baseline_life)
                                    for t in CHECKPOINTS}}
            folds.append(fold)
            print(domain, name, 'MAE',
                  round(fold['all_eligible']['cell_macro_mae_cycles'], 1), flush=True)
        summary = {'equal_fold_macro_mae_cycles': float(np.mean(
                       [f['all_eligible']['cell_macro_mae_cycles'] for f in folds])),
                   'worst_cell_mae_cycles': float(max(
                       f['all_eligible']['worst_cell_mae_cycles'] for f in folds)),
                   'checkpoints': {}}
        for t in CHECKPOINTS:
            present = [f['checkpoints'][str(t)] for f in folds
                       if f['checkpoints'][str(t)]['cells']]
            cell_rows = [r for score in present for r in score['by_cell']]
            summary['checkpoints'][str(t)] = {
                'cells': len(cell_rows), 'folds_with_coverage': len(present),
                'cell_macro_mae_cycles': float(np.mean(
                    [r['absolute_error_cycles'] for r in cell_rows])) if cell_rows else None,
                'cell_macro_life_mape': float(np.mean(
                    [r['life_absolute_percentage_error'] for r in cell_rows])) if cell_rows else None,
                'median_life_baseline_mae_cycles': float(np.mean(
                    [r['median_life_baseline_error_cycles'] for r in cell_rows]))
                    if cell_rows else None}
        report['domains'][domain] = {'summary': summary, 'folds': folds}
        if save_models:
            ids = [key for key, cell in cells.items() if cell.dataset == domain and
                   cell.endpoint_cycle is not None]
            OUT.mkdir(parents=True, exist_ok=True)
            model_name = (f'{domain.lower()}_experimental.joblib' if target == 'raw_eol'
                          else f'{domain.lower()}_{target}_experimental.joblib')
            joblib.dump(fit(cells, records, ids, features, target),
                        OUT / model_name)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=('NASA', 'CALCE', 'MIT', 'XJTU'),
                        default=['NASA', 'CALCE', 'MIT', 'XJTU'])
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--target', choices=('raw_eol', 'log_rul'), default='raw_eol')
    args = parser.parse_args()
    result = evaluate(args.domains, args.save_models, args.target)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = '' if set(args.domains) == {'NASA', 'CALCE', 'MIT', 'XJTU'} else \
        '_' + '-'.join(args.domains).lower()
    filename = 'development_cv' if args.target == 'raw_eol' else 'development_cv_log_rul'
    path = OUT / f'{filename}{suffix}.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', path, flush=True)
