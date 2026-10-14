"""Few-shot target-batch SOH calibration with disjoint calibration cells.

The zero-shot model remains the existing whole-batch MIT age/ExtraTrees OOF
model. For each target batch, one or two *other* cells with complete measured
SOH curves provide a scalar batch offset. Those cells are excluded from the
reported test metric. This is a separate adaptation regime, not zero-shot.
"""

import itertools
import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest


SOURCE = Path('battery_models_v2/soh_age_signal_challenger/mit_development_oof_predictions.npz')
OUT = Path('battery_models_v2/mit_physics_da_challenger/target_cell_calibration_audit.json')
SEED = 2026


def cell_data():
    groups = {row['cell_id']: row['group'] for row in load_manifest()['cells']
              if row['dataset'] == 'MIT'}
    with np.load(SOURCE) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    result = {}
    for key in np.unique(data['cell_id']):
        mask = (data['cell_id'] == key) & data['soh_available']
        target = np.asarray(data['target'][mask], dtype=float)
        estimate = np.asarray(data['gated_with_age'][mask], dtype=float)
        if len(target) < 20 or not np.isfinite(target).all():
            raise ValueError(f'{key}: invalid SOH calibration data')
        result[str(key)] = {'group': groups[str(key)], 'target': target,
                            'estimate': estimate}
    return result


def evaluate_calibration(cells, calibration_ids, heldout_ids, strength):
    if not calibration_ids or set(calibration_ids) & set(heldout_ids):
        raise ValueError('calibration and evaluation cells must be disjoint')
    correction = strength * float(np.mean([
        np.mean(cells[key]['target']-cells[key]['estimate'])
        for key in calibration_ids]))
    errors = []
    for key in heldout_ids:
        cell = cells[key]
        errors.append(float(np.mean(100*np.abs(
            cell['estimate']+correction-cell['target'])/cell['target'])))
    return float(np.mean(errors)), correction


def sampled_pairs(keys, count=25):
    pairs = list(itertools.combinations(keys, 2))
    rng = np.random.default_rng(SEED)
    positions = rng.choice(len(pairs), size=min(count, len(pairs)), replace=False)
    return [pairs[index] for index in positions]


def summarize_scenarios(cells, choices, keys):
    rows = []
    for calibration in choices:
        heldout = [key for key in keys if key not in calibration]
        baseline, _ = evaluate_calibration(cells, calibration, heldout, 0.)
        half, correction_half = evaluate_calibration(cells, calibration, heldout, .5)
        full, correction_full = evaluate_calibration(cells, calibration, heldout, 1.)
        rows.append({'calibration_cells': list(calibration),
                     'evaluated_cells': len(heldout),
                     'zero_shot_mape_percent': baseline,
                     'half_correction_mape_percent': half,
                     'full_correction_mape_percent': full,
                     'half_correction_soh': correction_half,
                     'full_correction_soh': correction_full})
    return {
        'calibration_sets': len(rows),
        'mean_zero_shot_mape_percent_on_matching_cells': float(np.mean([
            row['zero_shot_mape_percent'] for row in rows])),
        'mean_half_correction_mape_percent': float(np.mean([
            row['half_correction_mape_percent'] for row in rows])),
        'mean_full_correction_mape_percent': float(np.mean([
            row['full_correction_mape_percent'] for row in rows])),
        'p90_full_correction_mape_percent_across_calibrators': float(np.percentile([
            row['full_correction_mape_percent'] for row in rows], 90)),
        'by_calibration_set': rows,
    }


def main():
    cells = cell_data()
    report = {
        'status': 'development-domain adaptation diagnostic, not independent test',
        'base_model': 'whole-batch held-out MIT 300 s age/ExtraTrees OOF',
        'calibration': 'scalar SOH offset from full measured life of other '
                       'target-batch cell(s); no evaluation-cell labels',
        'warning': 'requires fully labeled target-batch calibration cells; '
                   'not zero-shot and not the online signal-only input setting',
        'batches': {},
    }
    for group in sorted({cell['group'] for cell in cells.values()}):
        keys = sorted(key for key, cell in cells.items() if cell['group'] == group)
        one = summarize_scenarios(cells, [(key,) for key in keys], keys)
        two = summarize_scenarios(cells, sampled_pairs(keys), keys)
        report['batches'][group] = {'cells': len(keys),
                                    'one_calibration_cell': one,
                                    'two_calibration_cells': two}
        print(group, '1-cell', {name: round(one[name], 3) for name in (
            'mean_zero_shot_mape_percent_on_matching_cells',
            'mean_half_correction_mape_percent',
            'mean_full_correction_mape_percent')},
              '2-cell', {name: round(two[name], 3) for name in (
            'mean_zero_shot_mape_percent_on_matching_cells',
            'mean_half_correction_mape_percent',
            'mean_full_correction_mape_percent')}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
