"""MIT RUL challenger with known, pre-cycle fast-charge protocol features.

Whole-batch CV is used. End-of-life labels and future capacity never enter
features. Censored cells are excluded from regression targets and from scores.
No model is promoted based solely on this development evaluation.
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from evaluate_paper_inspired_rul_v2 import score
from train_rul_v2 import eligible, signal_features
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/mit_policy_rul_challenger')
KINDS = ('signal_only', 'signal_policy')


def policy_values(record):
    """Use only a cell's declared charging schedule, not its batch ID."""
    steps = record.get('charge_rate_steps_c') or []
    protocol = record.get('charge_protocol') or ''
    fraction_match = re.search(r'\((\d+(?:\.\d+)?)%\)', protocol)
    fraction = float(fraction_match.group(1)) / 100 if fraction_match else np.nan
    first = float(steps[0]) if steps else np.nan
    second = float(steps[1]) if len(steps) >= 2 else np.nan
    # A few source policy strings omit the trailing C on the second stage.
    if not np.isfinite(second):
        second_match = re.search(r'\)-(\d+(?:\.\d+)?)C?', protocol)
        if second_match:
            second = float(second_match.group(1))
    if not np.isfinite(second) and len(steps) == 1:
        second = first
    weighted = (fraction * first + (1 - fraction) * second
                if np.isfinite(first) and np.isfinite(second) and np.isfinite(fraction)
                else np.nan)
    return np.asarray([first, second, fraction, weighted, second-first], dtype='float32')


def features(cell, record, kind):
    signal = signal_features(cell)
    if kind == 'signal_only':
        return signal
    values = policy_values(record)
    return np.column_stack((signal, np.tile(values, (len(cell.cycle), 1))))


def fit(cells, matrices, ids, kind):
    parts, targets, weights = [], [], []
    active = [key for key in ids if cells[key].endpoint_cycle is not None]
    for key in active:
        cell = cells[key]
        rows = np.flatnonzero(eligible(cell))[::5]
        if not len(rows):
            continue
        weight = np.where(cell.capacity_review_flag[rows], .25, 1.).astype('float64')
        weight /= weight.sum()
        parts.append(matrices[key][rows])
        targets.append(np.log1p(cell.rul_cycles[rows]))
        weights.append(weight / len(active))
    x, y, w = map(np.concatenate, (parts, targets, weights))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=w)
    return {'kind': kind, 'imputer': imputer, 'model': model,
            'training_cell_ids': active, 'data_contract_version': 2}


def predict(bundle, matrix):
    return np.maximum(0, np.expm1(bundle['model'].predict(
        bundle['imputer'].transform(matrix))))


def evaluate(save_candidate=True):
    manifest, all_cells = load_all()
    records = [row for row in manifest['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: all_cells[row['cell_id']] for row in records}
    record_by_id = {row['cell_id']: row for row in records}
    matrices = {kind: {key: features(cell, record_by_id[key], kind)
                       for key, cell in cells.items()} for kind in KINDS}
    folds = []
    for held, train_ids, val_ids in outer_splits(records, 'MIT'):
        if set(train_ids) & set(val_ids):
            raise ValueError('training/validation overlap')
        observed_val = [key for key in val_ids if cells[key].endpoint_cycle is not None]
        row = {'held_out': held, 'training_cells': len(train_ids),
               'validation_cells_with_endpoint': len(observed_val)}
        for kind in KINDS:
            fitted = fit(cells, matrices[kind], train_ids, kind)
            estimates = {key: predict(fitted, matrices[kind][key]) for key in observed_val}
            row[kind] = score(cells, observed_val, estimates)
        folds.append(row)
        print(held, {kind: {'mae': round(row[kind]['cell_macro_mae_cycles'], 1),
                            'eol_mape_at_50': round(row[kind]['eol_mape_at_50_percent'], 2)}
                     for kind in KINDS}, flush=True)
    summary = {}
    for kind in KINDS:
        summary[kind] = {'cell_macro_mae_cycles': float(np.mean([
            cell['mae_cycles'] for fold in folds for cell in fold[kind]['by_cell']])),
            'eol_mape_at_50_percent': float(np.mean([
                cell['eol_mape_at_50_percent'] for fold in folds
                for cell in fold[kind]['by_cell']
                if cell['eol_mape_at_50_percent'] is not None])),
            'cells_at_50': sum(fold[kind]['cells_at_50'] for fold in folds)}
    report = {'status': 'development whole-batch CV, not independent test',
              'policy': 'predeclared charge rates and transition fraction only',
              'censoring': '12 MIT censored cells excluded from supervised targets',
              'summary': summary, 'folds': folds}
    if save_candidate:
        OUT.mkdir(parents=True, exist_ok=True)
        ids = [key for key, cell in cells.items() if cell.endpoint_cycle is not None]
        joblib.dump(fit(cells, matrices['signal_policy'], ids, 'signal_policy'),
                    OUT / 'mit_policy_candidate.joblib')
        (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    evaluate()
