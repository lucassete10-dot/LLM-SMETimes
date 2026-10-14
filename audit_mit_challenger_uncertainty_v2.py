"""Cell-level uncertainty audit for MIT development-fold challenger results.

Stratified resampling keeps each held-out batch's cell count fixed. This is
descriptive uncertainty on reused development folds, NOT independent testing.
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path('battery_models_v2/mit_fractional_life_challenger')
SEED = 20260927
N_BOOTSTRAP = 10000


def stratified_difference(rows, baseline, challenger):
    groups = {}
    for row in rows:
        groups.setdefault(row['group'], []).append(
            float(baseline(row)-challenger(row)))
    values = [np.asarray(group, dtype=float) for group in groups.values()]
    if any(not len(group) for group in values):
        raise ValueError('A batch has no scored cells')
    rng = np.random.default_rng(SEED)
    samples = np.zeros(N_BOOTSTRAP, dtype=float)
    total = sum(len(group) for group in values)
    for group in values:
        indices = rng.integers(0, len(group), size=(N_BOOTSTRAP, len(group)))
        samples += group[indices].sum(axis=1)/total
    return {'scored_cells': total,
            'improvement_percentage_points': float(sum(group.sum() for group in values)/total),
            'cell_bootstrap_95pct_interval': [float(value) for value in
                np.quantile(samples, [.025, .975])],
            'batch_improvement_percentage_points': {
                name: float(np.mean(group)) for name, group in groups.items()}}


def main():
    age = json.loads((ROOT / 'age_blend_audit.json').read_text(encoding='utf-8'))
    curve = json.loads((ROOT / 'joint_curve_audit.json').read_text(encoding='utf-8'))
    curve_rows = [row for fold in curve['folds'] for row in fold['by_cell']]
    report = {
        'status': 'MIT development-fold descriptive uncertainty; not independent validation',
        'interpretation_limit': ('Only three MIT batches; cell bootstrap conditional on these '
                                 'batches does not measure transfer uncertainty to a new batch.'),
        'resampling': f'{N_BOOTSTRAP} cell-level draws, stratified by batch, seed {SEED}',
        'positive_improvement_means_challenger_better': True,
        'endpoint_origin_direct_minus_age_blend': stratified_difference(
            age['by_cell'], lambda row: row['selected_direct_origin_ape_percent'],
            lambda row: row['blend_origin_ape_percent']),
        'future_curve_direct_linear_minus_direct_power': stratified_difference(
            curve_rows, lambda row: row['metrics']['direct']['linear_curve_mape_percent'],
            lambda row: row['metrics']['direct']['power_curve_mape_percent']),
        'future_curve_direct_linear_minus_blend_power': stratified_difference(
            curve_rows, lambda row: row['metrics']['direct']['linear_curve_mape_percent'],
            lambda row: row['metrics']['blend']['power_curve_mape_percent']),
    }
    path = ROOT / 'uncertainty_audit.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', path)
    for key, result in report.items():
        if isinstance(result, dict):
            print(key, result)


if __name__ == '__main__':
    main()
