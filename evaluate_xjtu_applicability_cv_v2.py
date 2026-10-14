"""Check whether XJTU early-signal distance tracks held-out SOH/RUL error.

Every held-out strategy gets a fresh distance reference fitted only on the
other strategies. The per-cell error comes from already-saved whole-strategy
OOF predictions; no held-out capacity enters the signature or distance fit.
This is descriptive development evidence, not a calibrated rejection rule.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_manifest
from train_soh_v2 import outer_splits
from xjtu_applicability_v2 import signature


OUT = Path('battery_models_v2/xjtu_applicability_group_cv.json')


def fit_distance_reference(signatures):
    """Mirror the frozen distance recipe without seeing the held-out group."""
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(signatures)
    filled = imputer.transform(signatures)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(filled)
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    reference = np.clip(scaler.transform(filled), -10, 10)
    pairwise = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    p95 = float(np.quantile(np.min(pairwise, axis=1), .95))
    if not np.isfinite(p95) or p95 <= 0:
        raise ValueError('training-cell nearest-neighbor radius must be positive')
    return imputer, scaler, reference, p95


def distance_ratio(vector, fitted):
    imputer, scaler, reference, p95 = fitted
    filled = np.where(np.isnan(vector), imputer.statistics_, vector)
    scaled = np.clip((filled-scaler.center_)/scaler.scale_, -10, 10)
    return float(np.min(np.linalg.norm(reference-scaled[None, :], axis=1))/p95)


def error_lookup(path, metric):
    folds = json.loads(Path(path).read_text(encoding='utf-8'))
    result = {}
    for fold in folds:
        group = fold['held_out']
        for row in fold[metric]['by_cell']:
            key = row['cell_id']
            if key in result:
                raise ValueError(f'duplicate OOF error for {key}')
            result[key] = (group, row)
    return result


def summarize(rows):
    if not rows:
        return {'cells': 0, 'soh_mae': None, 'rul_mae_cycles': None}
    return {'cells': len(rows),
            'soh_mae': float(np.mean([row['soh_mae'] for row in rows])),
            'rul_mae_cycles': float(np.mean([row['rul_mae_cycles'] for row in rows]))}


def evaluate():
    records = load_manifest()['cells']
    xjtu = [row for row in records if row['dataset'] == 'XJTU']
    if len(xjtu) != 55:
        raise ValueError('expected audited 55-cell XJTU cohort')
    signatures = {}
    for row in xjtu:
        with np.load(row['feature_source'], allow_pickle=False) as archive:
            signatures[row['cell_id']] = signature(archive[row['feature_array']])
    soh = error_lookup('battery_models_v2/soh/xjtu/group_cv.json',
                       'extra_trees_history')
    rul = error_lookup('battery_models_v2/rul/xjtu/group_cv.json', 'direct')
    if set(soh) != set(signatures) or set(rul) != set(signatures):
        raise ValueError('stored OOF scores do not cover the audited XJTU cells')
    rows = []
    folds = []
    for group, train_ids, val_ids in outer_splits(records, 'XJTU'):
        training = np.stack([signatures[key] for key in train_ids])
        fitted = fit_distance_reference(training)
        held = []
        for key in val_ids:
            if soh[key][0] != group or rul[key][0] != group:
                raise ValueError(f'{key}: OOF score group mismatch')
            ratio = distance_ratio(signatures[key], fitted)
            item = {'cell_id': key, 'held_out': group, 'distance_ratio': ratio,
                    'outside_training_neighborhood': ratio > 1,
                    'soh_mae': soh[key][1]['mae'],
                    'rul_mae_cycles': rul[key][1]['mae_cycles']}
            rows.append(item)
            held.append(item)
        folds.append({'held_out': group, 'training_cells': len(train_ids),
                      'validation_cells': len(val_ids),
                      'outside_cells': sum(row['outside_training_neighborhood']
                                           for row in held),
                      'distance_ratio_median': float(np.median(
                          [row['distance_ratio'] for row in held])),
                      'within_group_spearman': {target: float(spearmanr(
                          [row['distance_ratio'] for row in held],
                          [row[target] for row in held]).statistic)
                          for target in ('soh_mae', 'rul_mae_cycles')},
                      'errors': summarize(held)})
    outside = [row for row in rows if row['outside_training_neighborhood']]
    inside = [row for row in rows if not row['outside_training_neighborhood']]
    correlation = {target: float(spearmanr(
        [row['distance_ratio'] for row in rows],
        [row[target] for row in rows]).statistic)
        for target in ('soh_mae', 'rul_mae_cycles')}
    group_correlation = {target: float(spearmanr(
        [fold['distance_ratio_median'] for fold in folds],
        [fold['errors'][target] for fold in folds]).statistic)
        for target in ('soh_mae', 'rul_mae_cycles')}
    return {'status': 'development-only descriptive whole-strategy OOF audit',
            'warning': 'distance is not calibrated as an error or abstention probability; '
                       'cells within a strategy are correlated and 55 cells are not '
                       '55 independent strategy tests',
            'distance_fit': 'only other strategy cells; first 20 label-free diagnostics',
            'summary': {'all': summarize(rows), 'outside': summarize(outside),
                        'within': summarize(inside),
                        'cell_level_spearman': correlation,
                        'group_level_spearman': group_correlation},
            'folds': folds, 'cells': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--verify', action='store_true')
    actions.add_argument('--replace', action='store_true',
                         help='regenerate this development-only audit')
    args = parser.parse_args()
    current = evaluate()
    if args.verify:
        if current != json.loads(OUT.read_text(encoding='utf-8')):
            raise ValueError('saved applicability CV audit differs from current inputs')
        print('VERIFIED', OUT)
    else:
        if OUT.exists() and not args.replace:
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print('WROTE', OUT)
