"""Coherent MIT future-curve audit from cycle-100 log-life endpoint estimates.

All SOH predictions are existing whole-batch out-of-fold values. Curve power
is fitted only on the two outer training batches for each held-out batch.
"""

import json
from pathlib import Path

import numpy as np

from audit_xjtu_blended_joint_curve_v2 import coherent_curve
from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_trained_knee_v2 import fit_power, power_curve
from train_soh_v2 import outer_splits
from train_trajectory_v2 import threshold_soh


ROOT = Path('battery_models_v2/mit_inverse_life_challenger')
CHECKPOINT = 100
SOH = Path('battery_models_v2/paper_inspired_soh_challenger/mit_development_oof_predictions.npz')


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    checkpoint = json.loads((ROOT / 'development_cv.json').read_text(encoding='utf-8'))
    predicted = {row['cell_id']: row for fold in checkpoint['folds'] for row in
                 fold['checkpoints'][str(CHECKPOINT)]['by_cell']}
    with np.load(SOH) as source:
        soh = {(str(key), int(cycle)): float(value)
               for key, cycle, value in zip(source['cell_id'], source['cycle'],
                                            source['gated'])}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        power, fitted = fit_power(cells, [key for key in train
                                         if cells[key].endpoint_cycle is not None])
        rows = []
        for key in validation:
            if key not in predicted:
                continue
            cell = cells[key]
            record = predicted[key]
            origin = record['observed_cycle']
            if (key, origin) not in soh:
                raise ValueError(f'missing held-out SOH: {key}, {origin}')
            future = np.flatnonzero((cell.cycle > origin) &
                                    (cell.cycle <= cell.endpoint_cycle) &
                                    cell.soh_available)
            if len(future) < 2:
                continue
            current = soh[key, origin]
            threshold = threshold_soh(cell)
            horizon = cell.cycle[future]-origin
            true = cell.soh[future]
            metrics = {}
            for name in ('baseline', 'log'):
                endpoint = record['predicted_endpoint_cycle'][name]
                linear = coherent_curve(current, threshold, endpoint-origin, horizon)
                shaped = power_curve(current, threshold, endpoint-origin, horizon, power)
                metrics[name] = {
                    'linear_curve_mape_percent': float(100*np.mean(abs(linear-true)/true)),
                    'power_curve_mape_percent': float(100*np.mean(abs(shaped-true)/true)),
                    'eol_ape_percent': record['ape_percent'][name]}
            rows.append({'cell_id': key, 'group': group, 'origin_cycle': origin,
                         'future_points': len(future), 'metrics': metrics})
        folds.append({'held_out': group, 'fitted_power': power,
                      'training_event_cells': fitted, 'scored_cells': len(rows),
                      'by_cell': rows})
        print(group, 'n', len(rows), 'power', round(power, 2), flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    summary = {'scored_cells': len(rows)}
    for name in ('baseline', 'log'):
        summary[name] = {metric: float(np.mean([row['metrics'][name][metric]
                                               for row in rows]))
                         for metric in ('linear_curve_mape_percent',
                                        'power_curve_mape_percent', 'eol_ape_percent')}
    report = {'status': 'unpromoted MIT whole-batch cycle-100 curve audit',
              'origin': 'last observed cycle at or before 100; at least 98',
              'endpoint': 'MIT source 0.88 Ah, not exact 80% SOH',
              'summary': summary, 'folds': folds}
    (ROOT / 'checkpoint_100_curve_audit.json').write_text(json.dumps(report, indent=2),
                                                           encoding='utf-8')
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
