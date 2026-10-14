"""Development-only test of explicit protocol metadata for held-out groups.

Only documented numeric operating conditions are used. A batch or strategy ID
is never used as a feature; missing values stay explicit. This is not deployed
unless cross-group results justify it and inference can require the same inputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from train_soh_v2 import fit as fit_baseline, outer_splits, predict as predict_baseline, score


OUT = Path('battery_models_v2/condition_feature_diagnostic.json')
NAMES = ('charge_rate_c', 'discharge_rate_c', 'diagnostic_discharge_rate_c',
         'ambient_temperature_c', 'charge_step_1_c', 'charge_step_2_c',
         'charge_step_3_c')


def condition_vector(record):
    steps = list(record.get('charge_rate_steps_c') or [])
    values = [record.get('charge_rate_c'), record.get('discharge_rate_c'),
              record.get('diagnostic_discharge_rate_c'),
              record.get('ambient_temperature_c'),
              *(steps[:3] + [None] * max(0, 3 - len(steps)))]
    numeric = np.array([np.nan if value is None else value for value in values],
                       dtype='float32')
    if len(numeric) != len(NAMES) or np.isinf(numeric).any():
        raise ValueError('invalid operating-condition metadata')
    return np.concatenate((numeric, np.isnan(numeric).astype('float32')))


def matrix(cell, metadata):
    condition = condition_vector(metadata)
    return np.concatenate((cell.matrix('signal_history'),
                           np.broadcast_to(condition, (len(cell.cycle), len(condition)))), axis=1)


def fit_condition(cells, records, ids):
    features, labels, weights = [], [], []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        if not keep.any():
            continue
        features.append(matrix(cell, records[key])[keep])
        labels.append(cell.soh[keep])
        w = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        weights.append(w / w.sum() / len(ids))
    x = np.concatenate(features)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), np.concatenate(labels),
              sample_weight=np.concatenate(weights))
    return imputer, model


def evaluate_domain(manifest, cells, domain):
    records = {row['cell_id']: row for row in manifest['cells']}
    folds = []
    for group, train_ids, val_ids in outer_splits(manifest['cells'], domain):
        baseline = fit_baseline(cells, train_ids, 'extra_trees_history')
        imputer, model = fit_condition(cells, records, train_ids)
        baseline_prediction = {key: predict_baseline(baseline, cells[key]) for key in val_ids}
        condition_prediction = {key: model.predict(imputer.transform(matrix(cells[key], records[key])))
                                for key in val_ids}
        folds.append({'held_out': group, 'baseline': score(cells, baseline_prediction, val_ids),
                      'with_conditions': score(cells, condition_prediction, val_ids)})
        print(domain, group, round(folds[-1]['baseline']['cell_macro_mae'], 5),
              round(folds[-1]['with_conditions']['cell_macro_mae'], 5), flush=True)
    return {'folds': folds, 'summary': {name: {
        'equal_fold_macro_mae': float(np.mean([row[name]['cell_macro_mae'] for row in folds])),
        'equal_cell_macro_mae': float(np.mean([cell['mae'] for row in folds
                                               for cell in row[name]['by_cell']])),
        'worst_cell_mae': float(np.max([cell['mae'] for row in folds
                                        for cell in row[name]['by_cell']])),
    } for name in ('baseline', 'with_conditions')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=('MIT', 'XJTU'),
                        default=['MIT', 'XJTU'])
    args = parser.parse_args()
    manifest, cells = load_all()
    result = {'status': 'development diagnostic; not deployed',
              'condition_columns': list(NAMES) + [name + '_missing' for name in NAMES],
              'note': 'No batch ID, strategy name, capacity, or SOH label used as input',
              'domains': {domain: evaluate_domain(manifest, cells, domain)
                          for domain in args.domains}}
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
