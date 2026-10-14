"""Development-only SOH ablation of absolute vs initial-relative signal views.

Uses the original ExtraTrees hyperparameters, weights and outer cell/group
splits. No held-out capacity enters features, imputation or model fitting.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from train_soh_v2 import outer_splits, score


OUT = Path('battery_models_v2/relative_soh_feature_diagnostic.json')
VIEWS = ('raw_only', 'delta_only', 'delta_history')


def baseline_oof(cells, domain):
    path = Path('battery_models_v2/soh') / domain.lower() / 'development_oof_predictions.npz'
    with np.load(path) as data:
        rows = list(zip(data['cell_id'], data['cycle'], data['estimate']))
    predictions = {key: np.full(len(cell.cycle), np.nan, dtype='float64')
                   for key, cell in cells.items() if cell.dataset == domain}
    for key, cycle, estimate in rows:
        key = str(key)
        cell = cells[key]
        index = int(np.searchsorted(cell.cycle, cycle))
        if index >= len(cell.cycle) or cell.cycle[index] != cycle or np.isfinite(predictions[key][index]):
            raise ValueError(f'OOF SOH alignment mismatch: {key}, {cycle}')
        predictions[key][index] = estimate
    for key, values in predictions.items():
        if not np.isfinite(values[cells[key].soh_available]).all():
            raise ValueError(f'incomplete baseline OOF SOH: {key}')
    return predictions


def view_matrix(cell, view):
    if view == 'raw_only':
        return cell.raw
    if view == 'delta_only':
        return cell.delta
    if view == 'delta_history':
        return np.concatenate((cell.delta, cell.history), axis=1)
    raise ValueError(f'unknown feature view: {view}')


def fit_view(cells, ids, view):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        if not keep.any():
            continue
        x_parts.append(view_matrix(cell, view)[keep])
        y_parts.append(cell.soh[keep])
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        w_parts.append(weight / weight.sum() / len(ids))
    x, y, weight = map(np.concatenate, (x_parts, y_parts, w_parts))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weight)
    return imputer, model


def predict_view(bundle, cell, view):
    imputer, model = bundle
    return model.predict(imputer.transform(view_matrix(cell, view)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=('NASA', 'CALCE', 'MIT', 'XJTU'),
                        default=['MIT', 'XJTU'])
    args = parser.parse_args()
    manifest, cells = load_all()
    report = {'status': 'development ablation; frozen model unchanged',
              'estimator': 'same ExtraTrees settings and cell weights as selected SOH model',
              'views': list(VIEWS), 'domains': {}}
    for domain in args.domains:
        baseline = baseline_oof(cells, domain)
        prior = {row['held_out']: row for row in json.loads(
            (Path('battery_models_v2/soh') / domain.lower() / 'group_cv.json').read_text(
                encoding='utf-8'))}
        folds = []
        for group, training, validation in outer_splits(manifest['cells'], domain):
            candidate = {}
            raw_predictions = None
            for view in VIEWS:
                fitted = fit_view(cells, training, view)
                predictions = {key: predict_view(fitted, cells[key], view)
                               for key in validation}
                if view == 'raw_only':
                    raw_predictions = predictions
                candidate[view] = score(cells, predictions, validation)
            baseline_score = score(cells, baseline, validation)
            if not np.isclose(baseline_score['cell_macro_mae'],
                              prior[group]['extra_trees_history']['cell_macro_mae'], atol=1e-7):
                raise ValueError('stored baseline OOF differs from fold report')
            half = {key: .5 * (baseline[key] + raw_predictions[key]) for key in validation}
            candidate['baseline_raw_half'] = score(cells, half, validation)
            folds.append({'held_out': group,
                          'baseline_all': prior[group]['extra_trees_history'],
                          **candidate})
            print(domain, group, {view: round(folds[-1][view]['cell_macro_mae'], 5)
                                  for view in ('baseline_all', *VIEWS, 'baseline_raw_half')},
                  flush=True)
        summary = {view: {
            'equal_fold_macro_mae': float(np.mean([row[view]['cell_macro_mae'] for row in folds])),
            'equal_cell_macro_mae': float(np.mean([cell['mae'] for row in folds
                                                   for cell in row[view]['by_cell']])),
            'worst_cell_mae': float(np.max([cell['mae'] for row in folds
                                            for cell in row[view]['by_cell']])),
            'folds_improved_vs_baseline': sum(row[view]['cell_macro_mae'] <
                                              row['baseline_all']['cell_macro_mae'] for row in folds),
        } for view in ('baseline_all', *VIEWS, 'baseline_raw_half')}
        report['domains'][domain] = {'summary': summary, 'folds': folds}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
