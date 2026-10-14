"""Retrospective leave-split-out residual bands; not a coverage guarantee.

The point predictions are whole-cell/batch out of fold. For each held-out
development split, the radius comes only from errors in other splits. Because
model selection used these splits and calibration-fold models may have seen the
test split during their own fits, this is an empirical stress test, not formal
split conformal prediction. A genuinely independent test is still required.
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path('battery_models_v2')
DOMAINS = ('NASA', 'CALCE', 'MIT', 'XJTU')
TASKS = (('soh', 0.9), ('rul', 0.9))


def quantile_radius(errors, coverage):
    return float(np.quantile(np.abs(errors), coverage, method='higher'))


def cell_macro(values, cell_ids):
    return float(np.mean([np.mean(values[cell_ids == key]) for key in np.unique(cell_ids)]))


def evaluate(path, coverage):
    with np.load(path) as source:
        actual = source['target']
        estimate = source['estimate']
        split = source['held_out']
        cell = source['cell_id']
        cycle = source['cycle']
        endpoint = source['endpoint'] if 'endpoint' in source.files else None
    error = estimate-actual
    if not (np.isfinite(actual).all() and np.isfinite(estimate).all()):
        raise ValueError(f'{path}: nonfinite out-of-fold values')
    folds = []
    for name in np.unique(split):
        validation = split == name
        calibration = ~validation
        radius = quantile_radius(error[calibration], coverage)
        covered = np.abs(error[validation]) <= radius
        by_phase = {}
        if endpoint is not None:
            progress = cycle[validation] / endpoint[validation]
            masks = {'early': progress < .33,
                     'middle': (progress >= .33) & (progress < .67),
                     'late': progress >= .67}
            by_phase = {phase: {'observations': int(mask.sum()),
                                'coverage': float(np.mean(covered[mask])) if mask.any() else None}
                        for phase, mask in masks.items()}
        folds.append({'held_out': str(name), 'calibration_radius': radius,
                      'validation_cells': int(len(np.unique(cell[validation]))),
                      'validation_observations': int(validation.sum()),
                      'observation_coverage': float(np.mean(covered)),
                      'cell_macro_coverage': cell_macro(covered, cell[validation]),
                      'signed_bias': float(np.mean(error[validation])),
                      'by_phase': by_phase})
    return {'nominal_coverage': coverage,
            'development_oof_radius': quantile_radius(error, coverage),
            'observations': int(len(actual)), 'cells': int(len(np.unique(cell))),
            'cross_split_observation_coverage': float(np.mean(
                [fold['observation_coverage'] for fold in folds])),
            'cross_split_cell_macro_coverage': float(np.mean(
                [fold['cell_macro_coverage'] for fold in folds])),
            'worst_split_coverage': float(min(fold['observation_coverage'] for fold in folds)),
            'folds': folds,
            'warning': 'retrospective development diagnostic only; no finite-sample coverage guarantee'}


def main():
    report = {'data_contract_version': 2,
              'calibration_design': 'leave one development split out of the residual-radius calculation',
              'warning': 'not a final independent or formal conformal coverage result',
              'tasks': {}}
    for task, nominal in TASKS:
        report['tasks'][task] = {}
        for domain in DOMAINS:
            path = ROOT / task / domain.lower() / 'development_oof_predictions.npz'
            metric = evaluate(path, nominal)
            report['tasks'][task][domain] = metric
            print(task, domain, 'radius', round(metric['development_oof_radius'], 3),
                  'cross-split coverage', round(metric['cross_split_observation_coverage'], 3),
                  'worst', round(metric['worst_split_coverage'], 3), flush=True)
    (ROOT / 'development_intervals.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
