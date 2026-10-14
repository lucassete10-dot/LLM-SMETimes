"""XJTU RUL target-transform ablation under the frozen whole-strategy folds.

Only the regression target changes from log1p(RUL) to raw RUL cycles. Signals,
age, cell weights, downsampling, estimator, and observed-endpoint filtering
match train_rul_v2. This is post-hoc development work, not a new selected model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from train_rul_v2 import (assess, eligible, fit_rul, predict_rul, signal_features)
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/rul/xjtu/target_transform_diagnostic.json')
PRIOR = Path('battery_models_v2/rul/xjtu/group_cv.json')


def fit_raw_target(cells, ids):
    inputs, targets, weights = [], [], []
    for key in ids:
        cell = cells[key]
        keep = eligible(cell)
        if not keep.any():
            continue
        x = signal_features(cell)[keep]
        y = cell.rul_cycles[keep]
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        weight /= weight.sum()
        inputs.append(x[::5])
        targets.append(y[::5])
        weights.append(weight[::5]/len(ids))
    x, y, sample_weight = map(np.concatenate, (inputs, targets, weights))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=sample_weight)
    return imputer, model


def score_summary(folds, name):
    values = [fold[name] for fold in folds]
    return {
        'equal_fold_macro_mae_cycles': float(np.mean(
            [item['cell_macro_mae_cycles'] for item in values])),
        'equal_fold_macro_signed_bias_cycles': float(np.mean(
            [item['cell_macro_signed_bias_cycles'] for item in values])),
        'worst_cell_mae_cycles': float(max(row['mae_cycles'] for item in values
                                           for row in item['by_cell'])),
        'phase_equal_fold_macro_mae_cycles': {phase: float(np.mean([
            item['phase_macro_mae_cycles'][phase] for item in values
            if item['phase_macro_mae_cycles'][phase] is not None]))
            for phase in ('early', 'middle', 'late')},
    }


def evaluate():
    manifest, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'XJTU'}
    stored = {fold['held_out']: fold['direct'] for fold in
              json.loads(PRIOR.read_text(encoding='utf-8'))}
    folds = []
    for group, training, validation in outer_splits(manifest['cells'], 'XJTU'):
        train = [key for key in training if cells[key].endpoint_cycle is not None]
        val = [key for key in validation if cells[key].endpoint_cycle is not None]
        median_life = float(np.median([cells[key].endpoint_cycle for key in train]))
        log_model = fit_rul(cells, train, 'direct')
        log_predictions = {key: predict_rul(log_model, cells[key]) for key in val}
        log_score = assess(cells, val, log_predictions, median_life)
        if not np.isclose(log_score['cell_macro_mae_cycles'],
                          stored[group]['cell_macro_mae_cycles'], atol=1e-6):
            raise ValueError(f'{group}: recomputed frozen baseline differs from stored CV')
        imputer, raw_model = fit_raw_target(cells, train)
        raw_predictions = {key: np.maximum(0, raw_model.predict(
            imputer.transform(signal_features(cells[key])))) for key in val}
        raw_score = assess(cells, val, raw_predictions, median_life)
        folds.append({'held_out': group, 'log1p_target': log_score,
                      'raw_cycle_target': raw_score})
        print(group, 'log', round(log_score['cell_macro_mae_cycles'], 1),
              'raw', round(raw_score['cell_macro_mae_cycles'], 1), flush=True)
    summary = {name: score_summary(folds, name)
               for name in ('log1p_target', 'raw_cycle_target')}
    summary['raw_improved_folds'] = sum(
        fold['raw_cycle_target']['cell_macro_mae_cycles'] <
        fold['log1p_target']['cell_macro_mae_cycles'] for fold in folds)
    return {'status': 'post-hoc development target-transform diagnostic; no model deployed',
            'comparison': 'same 55 cells, observed endpoints, causal signal/age inputs, '
                          'weights, downsampling, ExtraTrees settings, six held-out strategies',
            'warning': 'development-fold choice is not independent evidence; future '
                       'strategy still cannot be inferred from current signals alone',
            'summary': summary, 'folds': folds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    current = evaluate()
    if args.verify:
        stored = json.loads(OUT.read_text(encoding='utf-8'))
        for name in ('log1p_target', 'raw_cycle_target'):
            for key, value in current['summary'][name].items():
                if isinstance(value, dict):
                    for phase, score in value.items():
                        if not np.isclose(score, stored['summary'][name][key][phase],
                                          atol=1e-8, rtol=1e-10):
                            raise ValueError(f'{name}.{key}.{phase} differs')
                elif not np.isclose(value, stored['summary'][name][key],
                                    atol=1e-8, rtol=1e-10):
                    raise ValueError(f'{name}.{key} differs')
        if current['summary']['raw_improved_folds'] != stored['summary']['raw_improved_folds']:
            raise ValueError('improved-fold count differs')
        print('REPRODUCED', OUT)
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print('WROTE', OUT)
