"""Nested, whole-strategy SOH interval audit for 120/300-second XJTU inputs.

For each outer validation strategy, calibrate absolute-error radii from inner
leave-strategy-out fits on the other five strategies only. This is a
development stress test, not a formal guarantee for unseen operating regimes.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from evaluate_interval_robustness_v2 import METHODS, cell_macro_coverage, radii
from evaluate_xjtu_long_prefix_v2 import fit, matrix
from train_soh_v2 import outer_splits, score


ROOT = Path('battery_models_v2/xjtu_long_prefix')
OUT = ROOT / 'nested_interval_diagnostic.json'


def inner_splits(cells, outer_train_ids):
    groups = sorted({cells[key].group for key in outer_train_ids})
    if len(groups) < 3:
        raise ValueError('nested calibration requires at least three training strategies')
    for held in groups:
        train = [key for key in outer_train_ids if cells[key].group != held]
        validation = [key for key in outer_train_ids if cells[key].group == held]
        if set(train) & set(validation) or set(train) | set(validation) != set(outer_train_ids):
            raise ValueError(f'{held}: invalid inner whole-strategy split')
        yield held, train, validation


def predict(model_bundle, features, ids):
    imputer, model = model_bundle
    return {key: model.predict(imputer.transform(features[key])) for key in ids}


def labeled_errors(cells, predictions, ids):
    errors, cell_ids, groups = [], [], []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        if not keep.any():
            raise ValueError(f'{key}: no scored SOH observations')
        error = predictions[key][keep] - cell.soh[keep]
        errors.append(error)
        cell_ids.extend([key] * len(error))
        groups.extend([cell.group] * len(error))
    return np.concatenate(errors), np.asarray(cell_ids), np.asarray(groups)


def coverage(cells, predictions, ids, radius):
    hit_parts, id_parts = [], []
    by_phase = {'early': [], 'middle': [], 'late': []}
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        error = np.abs(predictions[key][keep] - cell.soh[keep])
        hit = error <= radius
        hit_parts.append(hit)
        id_parts.extend([key] * len(hit))
        if cell.endpoint_cycle is not None:
            progress = cell.cycle[keep] / cell.endpoint_cycle
            for name, mask in (('early', progress < .33),
                               ('middle', (progress >= .33) & (progress < .67)),
                               ('late', progress >= .67)):
                if mask.any():
                    by_phase[name].append(float(np.mean(hit[mask])))
    all_hits = np.concatenate(hit_parts)
    return {'radius': float(radius), 'full_width': float(2*radius),
            'observation_coverage': float(np.mean(all_hits)),
            'cell_macro_coverage': cell_macro_coverage(all_hits, np.asarray(id_parts)),
            'phase_cell_macro_coverage': {name: float(np.mean(values)) if values else None
                                          for name, values in by_phase.items()}}


def features_by_window(records, cells):
    result = {120: {key: cells[key].matrix('signal_history') for key in cells}}
    with np.load(ROOT / 'features.npz') as archive:
        result[300] = {}
        for row in records:
            key = row['cell_id']
            if not np.array_equal(archive[f'{key}_cycle'], cells[key].cycle):
                raise ValueError(f'{key}: long-prefix/source cycle mismatch')
            result[300][key] = matrix(archive[f'{key}_300s'].astype('float32'),
                                      cells[key].reference_count)
    return result


def evaluate():
    manifest, loaded = load_all()
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    cells = {row['cell_id']: loaded[row['cell_id']] for row in records}
    features = features_by_window(records, cells)
    prior = {row['held_out']: row for row in json.loads(
        (ROOT / 'group_cv.json').read_text(encoding='utf-8'))['folds']}
    folds = []
    for held, train_ids, val_ids in outer_splits(records, 'XJTU'):
        row = {'held_out': held, 'outer_train_cell_ids': train_ids,
               'outer_validation_cell_ids': val_ids, 'windows': {}}
        if {cells[key].group for key in train_ids} & {held}:
            raise ValueError(f'{held}: validation strategy in outer training')
        for window in (120, 300):
            feature = features[window]
            train_predictions, calibration_ids = {}, []
            for inner_held, inner_train, inner_val in inner_splits(cells, train_ids):
                if inner_held == held:
                    raise ValueError('outer validation strategy entered inner calibration')
                inner_pred = predict(fit(cells, feature, inner_train), feature, inner_val)
                train_predictions.update(inner_pred)
                calibration_ids.extend(inner_val)
            if set(train_predictions) != set(train_ids) or len(calibration_ids) != len(train_ids):
                raise ValueError(f'{held}/{window}: incomplete inner OOF predictions')
            inner_error, inner_cell, inner_group = labeled_errors(
                cells, train_predictions, calibration_ids)
            calibrated = radii(inner_error, inner_cell, inner_group)
            outer_pred = predict(fit(cells, feature, train_ids), feature, val_ids)
            metric = score(cells, outer_pred, val_ids)
            saved = prior[held]['120s_baseline' if window == 120 else '300s']
            if not np.isclose(metric['cell_macro_mae'], saved['cell_macro_mae'], atol=1e-9):
                raise ValueError(f'{held}/{window}: outer SOH fit differs from saved CV')
            row['windows'][f'{window}s'] = {
                'soh_cell_macro_mae': metric['cell_macro_mae'],
                'calibration_cells': len(train_ids),
                'calibration_groups': sorted(set(inner_group)),
                'calibration_observations': len(inner_error),
                'methods': {name: coverage(cells, outer_pred, val_ids, calibrated[name])
                            for name in METHODS},
            }
            print(held, window, 'MAE', round(metric['cell_macro_mae'], 5),
                  'cell-p90 coverage', round(row['windows'][f'{window}s']['methods']
                                               ['cell_p90_then_p90']['cell_macro_coverage'], 3),
                  flush=True)
        folds.append(row)
    summary = {}
    for window in ('120s', '300s'):
        summary[window] = {}
        for method in METHODS:
            values = [fold['windows'][window]['methods'][method] for fold in folds]
            summary[window][method] = {
                'equal_fold_cell_macro_coverage': float(np.mean(
                    [value['cell_macro_coverage'] for value in values])),
                'worst_fold_cell_macro_coverage': float(min(
                    value['cell_macro_coverage'] for value in values)),
                'equal_fold_radius': float(np.mean([value['radius'] for value in values])),
                'folds_at_or_above_90pct_cell_coverage': sum(
                    value['cell_macro_coverage'] >= .9 for value in values),
            }
    return {'status': 'development-only nested whole-strategy interval stress test',
            'calibration': 'each outer radius uses only inner OOF residuals from the other '
                           'five strategies; no validation labels enter model or radius',
            'warning': 'six development strategies and post-hoc 300-second feature choice '
                       'cannot support formal 90% coverage or independent accuracy claims',
            'summary': summary, 'folds': folds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    current = evaluate()
    if args.verify:
        if current != json.loads(OUT.read_text(encoding='utf-8')):
            raise ValueError('saved nested interval audit differs from current data')
        print('VERIFIED', OUT)
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print('WROTE', OUT)


if __name__ == '__main__':
    main()
