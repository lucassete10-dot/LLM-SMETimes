"""Development-only interval-width versus held-out-batch coverage audit.

All radii for a held-out batch are computed from other batches' OOF errors.
The 90th-of-cell-90th and worst-training-group alternatives are descriptive
stress tests, not formal conformal guarantees or independently selected bands.
"""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path('battery_models_v2')
OUT = ROOT / 'interval_robustness_diagnostic.json'
DOMAINS = ('MIT', 'XJTU')
TASKS = ('soh', 'rul')
METHODS = ('pooled_cycle_p90', 'cell_p90_then_p90', 'worst_training_group_p90')


def upper_quantile(values, q=.9):
    if not len(values):
        raise ValueError('interval calibration set is empty')
    return float(np.quantile(values, q, method='higher'))


def radii(errors, cell_ids, groups):
    absolute = np.abs(np.asarray(errors, dtype='float64'))
    cell_ids = np.asarray(cell_ids)
    groups = np.asarray(groups)
    if len(absolute) != len(cell_ids) or len(absolute) != len(groups):
        raise ValueError('interval calibration arrays differ in length')
    by_cell = [upper_quantile(absolute[cell_ids == key]) for key in np.unique(cell_ids)]
    by_group = [upper_quantile(absolute[groups == key]) for key in np.unique(groups)]
    return {'pooled_cycle_p90': upper_quantile(absolute),
            'cell_p90_then_p90': upper_quantile(by_cell),
            'worst_training_group_p90': float(max(by_group))}


def cell_macro_coverage(mask, cell_ids):
    return float(np.mean([np.mean(mask[cell_ids == key]) for key in np.unique(cell_ids)]))


def evaluate_one(task, domain):
    path = ROOT / task / domain.lower() / 'development_oof_predictions.npz'
    with np.load(path, allow_pickle=False) as archive:
        actual = archive['target']
        estimate = archive['estimate']
        group = archive['held_out']
        cell = archive['cell_id']
        cycle = archive['cycle']
        endpoint = archive['endpoint'] if 'endpoint' in archive.files else None
    if not (len(actual) == len(estimate) == len(group) == len(cell) == len(cycle)):
        raise ValueError(f'{task}/{domain}: OOF arrays do not align')
    error = estimate-actual
    if not np.isfinite(error).all():
        raise ValueError(f'{task}/{domain}: nonfinite OOF error')
    folds = []
    for held in np.unique(group):
        train = group != held
        val = ~train
        calibrated = radii(error[train], cell[train], group[train])
        methods = {}
        for name, radius in calibrated.items():
            covered = np.abs(error[val]) <= radius
            phase = {}
            if endpoint is not None:
                progress = cycle[val]/endpoint[val]
                phase_mask = {'early': progress < .33,
                              'middle': (progress >= .33) & (progress < .67),
                              'late': progress >= .67}
                phase = {key: float(np.mean(covered[mask])) if mask.any() else None
                         for key, mask in phase_mask.items()}
            methods[name] = {'radius': radius, 'full_width': 2*radius,
                             'observation_coverage': float(np.mean(covered)),
                             'cell_macro_coverage': cell_macro_coverage(covered, cell[val]),
                             'phase_coverage': phase}
        folds.append({'held_out': str(held), 'cells': int(len(np.unique(cell[val]))),
                      'observations': int(val.sum()), 'methods': methods})
    summary = {name: {
        'equal_fold_observation_coverage': float(np.mean([
            fold['methods'][name]['observation_coverage'] for fold in folds])),
        'equal_fold_cell_macro_coverage': float(np.mean([
            fold['methods'][name]['cell_macro_coverage'] for fold in folds])),
        'worst_fold_observation_coverage': float(min(
            fold['methods'][name]['observation_coverage'] for fold in folds)),
        'equal_fold_radius': float(np.mean([
            fold['methods'][name]['radius'] for fold in folds])),
        'folds_at_or_above_90pct': sum(
            fold['methods'][name]['observation_coverage'] >= .9 for fold in folds),
    } for name in METHODS}
    return {'summary': summary, 'folds': folds}


def evaluate():
    return {'status': 'retrospective development stress test; default bands unchanged',
            'warning': 'OOF models and feature choices used these development groups; '
                       'radii are not formal group-conformal or external guarantees. '
                       'Larger radii may be too wide to guide decisions.',
            'methods': list(METHODS),
            'tasks': {task: {domain: evaluate_one(task, domain) for domain in DOMAINS}
                      for task in TASKS}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    current = evaluate()
    if args.verify:
        if current != json.loads(OUT.read_text(encoding='utf-8')):
            raise ValueError('saved interval robustness report differs from OOF data')
        print('VERIFIED', OUT)
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print('WROTE', OUT)
