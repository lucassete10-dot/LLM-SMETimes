"""Development-only RUL ablation: SOH trend per source cycle, not per row.

XJTU diagnostic observations can be separated by many aging cycles. This
tests whether a 20-observation or 20-cycle SOH slope, computed with actual
source-cycle spans, improves whole-strategy RUL generalization.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from train_rul_v2 import assess, eligible, oof_soh, signal_features
from train_soh_v2 import fit as fit_soh, outer_splits, predict as predict_soh


OUT = Path('battery_models_v2/cycle_slope_rul_diagnostic.json')
MODES = ('last20_observations_per_cycle', 'last20_cycles')


def corrected_features(cell, soh_pred, mode):
    soh_pred = np.asarray(soh_pred, dtype='float32')
    if len(soh_pred) != len(cell.cycle):
        raise ValueError('SOH estimate must align with source cycles')
    cycle = np.asarray(cell.cycle, dtype='int64')
    if np.any(np.diff(cycle) <= 0):
        raise ValueError('source cycles must be strictly increasing')
    index = np.arange(len(cycle))
    if mode == 'last20_observations_per_cycle':
        lag = np.maximum(0, index - 20)
    elif mode == 'last20_cycles':
        lag = np.searchsorted(cycle, cycle - 20, side='left')
    else:
        raise ValueError(f'unknown slope mode: {mode}')
    span = np.maximum(1, cycle - cycle[lag])
    slope = (soh_pred - soh_pred[lag]) / span
    smooth = np.array([np.median(soh_pred[max(0, i - 4):i + 1])
                       for i in index], dtype='float32')
    return np.column_stack((signal_features(cell), soh_pred, smooth, slope)).astype('float32')


def fit_corrected(cells, ids, soh_predictions, mode):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        keep = eligible(cell)
        if not keep.any():
            continue
        x = corrected_features(cell, soh_predictions[key], mode)[keep]
        y = cell.rul_cycles[keep]
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        weight /= weight.sum()
        x_parts.append(x[::5])
        y_parts.append(y[::5])
        w_parts.append(weight[::5] / len(ids))
    x, y, weight = map(np.concatenate, (x_parts, y_parts, w_parts))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), np.log1p(y), sample_weight=weight)
    return imputer, model


def predict_corrected(bundle, cell, soh_prediction, mode):
    imputer, model = bundle
    x = corrected_features(cell, soh_prediction, mode)
    return np.maximum(0, np.expm1(model.predict(imputer.transform(x))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=('MIT', 'XJTU'),
                        default=['MIT', 'XJTU'])
    args = parser.parse_args()
    manifest, cells = load_all()
    report = {'status': 'development ablation; frozen selected models unchanged',
              'feature_change': 'only SOH slope denominator/window; other features, estimator and folds held fixed',
              'domains': {}}
    for domain in args.domains:
        folds = []
        baseline_folds = {row['held_out']: row for row in json.loads(
            (Path('battery_models_v2/rul') / domain.lower() / 'group_cv.json').read_text(
                encoding='utf-8'))}
        for group, training, validation in outer_splits(manifest['cells'], domain):
            observed_train = [key for key in training if cells[key].endpoint_cycle is not None]
            observed_val = [key for key in validation if cells[key].endpoint_cycle is not None]
            oof = oof_soh(cells, manifest['cells'], observed_train)
            soh_model = fit_soh(cells, observed_train, 'extra_trees_history')
            soh = {key: predict_soh(soh_model, cells[key]) for key in observed_val}
            base_life = float(np.median([cells[key].endpoint_cycle for key in observed_train]))
            candidates = {}
            for mode in MODES:
                model = fit_corrected(cells, observed_train, oof, mode)
                predictions = {key: predict_corrected(model, cells[key], soh[key], mode)
                               for key in observed_val}
                candidates[mode] = assess(cells, observed_val, predictions, base_life)
            baseline = baseline_folds[group]['assisted']
            folds.append({'held_out': group, 'baseline_assisted': baseline,
                          **candidates})
            print(domain, group, 'baseline', round(baseline['cell_macro_mae_cycles'], 1),
                  {mode: round(candidates[mode]['cell_macro_mae_cycles'], 1)
                   for mode in MODES}, flush=True)
        summary = {mode: {
            'equal_fold_macro_mae_cycles': float(np.mean([
                row[mode]['cell_macro_mae_cycles'] for row in folds])),
            'equal_cell_macro_mae_cycles': float(np.mean([
                cell['mae_cycles'] for row in folds for cell in row[mode]['by_cell']])),
            'worst_cell_mae_cycles': float(np.max([
                cell['mae_cycles'] for row in folds for cell in row[mode]['by_cell']])),
            'early_phase_macro_mae_cycles': float(np.mean([
                cell['by_phase']['early']['mae_cycles'] for row in folds
                for cell in row[mode]['by_cell']
                if cell['by_phase']['early']['mae_cycles'] is not None])),
        } for mode in ('baseline_assisted', *MODES)}
        report['domains'][domain] = {'summary': summary, 'folds': folds}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
