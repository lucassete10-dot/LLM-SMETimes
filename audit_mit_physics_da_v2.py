"""Inspect prediction quality and physical upward-jump violations of ablations."""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest


ROOT = Path('battery_models_v2/mit_physics_da_challenger')
PREDICTIONS = ROOT/'development_oof.npz'
BASELINE = Path('battery_models_v2/soh_age_signal_challenger/mit_development_oof_predictions.npz')
OUT = ROOT/'prediction_diagnostics.json'
VARIANTS = ('age_extratrees', 'tiny_mlp', 'physics_only',
            'coral_only', 'physics_and_coral')


def causal_upward_cap(cycle, estimate, regeneration_per_cycle=.002):
    """Bound each predicted rise using only current/past model outputs."""
    cycle = np.asarray(cycle, dtype=int)
    result = np.asarray(estimate, dtype=float).copy()
    if len(cycle) != len(result) or np.any(np.diff(cycle) <= 0):
        raise ValueError('predictions must follow increasing cycles')
    for index in range(1, len(result)):
        limit = result[index-1] + regeneration_per_cycle * (
            cycle[index]-cycle[index-1])
        result[index] = min(result[index], limit)
    return result


def score_one(cycle, true, available, prediction):
    keep = np.flatnonzero(available)
    target, estimate = true[keep], prediction[keep]
    ape = 100*np.abs(estimate-target)/target
    gap = np.diff(cycle[keep])
    excess = np.maximum(0., np.diff(estimate)-.002*gap)
    return {'mape_percent': float(np.mean(ape)),
            'p90_ape_percent': float(np.percentile(ape, 90)),
            'signed_bias_soh': float(np.mean(estimate-target)),
            'upward_violation_fraction': float(np.mean(excess > 1e-8)),
            'mean_upward_violation_soh': float(np.mean(excess))}


def main():
    with np.load(PREDICTIONS) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    with np.load(BASELINE) as archive:
        if not (np.array_equal(data['cell_id'], archive['cell_id'])
                and np.array_equal(data['cycle'], archive['cycle'])):
            raise ValueError('baseline and ablation OOF rows are not aligned')
        data['age_extratrees'] = archive['gated_with_age'].copy()
    groups = {row['cell_id']: row['group'] for row in load_manifest()['cells']
              if row['dataset'] == 'MIT'}
    rows = []
    for key in np.unique(data['cell_id']):
        mask = data['cell_id'] == key
        row = {'cell_id': str(key), 'group': groups[str(key)]}
        row['true'] = score_one(data['cycle'][mask], data['target'][mask],
                                data['soh_available'][mask], data['target'][mask])
        row['metrics'] = {}
        for name in VARIANTS:
            row['metrics'][name] = score_one(
                data['cycle'][mask], data['target'][mask],
                data['soh_available'][mask], data[name][mask])
            eligible = data['soh_available'][mask]
            capped = data[name][mask].astype(float, copy=True)
            capped[eligible] = causal_upward_cap(data['cycle'][mask][eligible],
                                                  capped[eligible])
            row['metrics'][name+'_capped'] = score_one(
                data['cycle'][mask], data['target'][mask], eligible, capped)
        rows.append(row)
    names = [variant for name in VARIANTS for variant in (name, name+'_capped')]
    summary = {name: {metric: float(np.mean([
        row['metrics'][name][metric] for row in rows]))
        for metric in ('mape_percent', 'p90_ape_percent', 'signed_bias_soh',
                       'upward_violation_fraction', 'mean_upward_violation_soh')}
        for name in names}
    report = {'status': 'held-out MIT development diagnostic, not independent test',
              'physics_rule': 'predicted SOH increase above .002 per source cycle',
              'true_upward_violation_fraction_cell_macro': float(np.mean([
                  row['true']['upward_violation_fraction'] for row in rows])),
              'summary': summary, 'by_cell': rows}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name in names:
        print(name, 'MAPE', round(summary[name]['mape_percent'], 3),
              'upward violations',
              round(summary[name]['upward_violation_fraction'], 3), flush=True)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
