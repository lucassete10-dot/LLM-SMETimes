"""MIT cycle-100 coherent SOH-curve audit using observed ΔQ(V) diagnostics.

The endpoint model uses only cycle 10/100 complete discharge Q(V). Current
SOH is an existing whole-batch OOF estimate from the 300 s prefix; the curve
power is fitted on other batches. This is diagnostic-assisted, not online.
"""

import json
from pathlib import Path

import numpy as np

from audit_xjtu_blended_joint_curve_v2 import coherent_curve
from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_trained_knee_v2 import fit_power, power_curve
from train_soh_v2 import outer_splits
from train_trajectory_v2 import threshold_soh


ROOT = Path('battery_models_v2/mit_qv_challenger')
SOH = Path('battery_models_v2/paper_inspired_soh_challenger/mit_development_oof_predictions.npz')
PRIOR = Path('battery_models_v2/mit_inverse_life_challenger/development_cv.json')
PAPER = ROOT / 'paper_discharge_cv.json'


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    stats = json.loads((ROOT / 'deltaq_stats_cv.json').read_text(encoding='utf-8'))
    endpoint = {row['cell_id']: row for fold in stats['folds'] for row in fold['by_cell']}
    paper = json.loads(PAPER.read_text(encoding='utf-8'))
    paper_endpoint = {row['cell_id']: row for fold in paper['folds']
                      for row in fold['by_cell']}
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    baseline = {row['cell_id']: row for fold in prior['folds'] for row in
                fold['checkpoints']['100']['by_cell']}
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
            if key not in endpoint or key not in baseline or key not in paper_endpoint:
                continue
            cell = cells[key]
            origin = baseline[key]['observed_cycle']
            if (key, origin) not in soh:
                raise ValueError(f'missing OOF SOH: {key}, {origin}')
            future = np.flatnonzero((cell.cycle > origin) &
                                    (cell.cycle <= cell.endpoint_cycle) &
                                    cell.soh_available)
            if len(future) < 2:
                continue
            current = soh[key, origin]
            threshold = threshold_soh(cell)
            horizons = cell.cycle[future]-origin
            true = cell.soh[future]
            eol = {'selected_direct': baseline[key]['predicted_endpoint_cycle']['baseline'],
                   'deltaq_variance_min': endpoint[key]['predicted_endpoint_cycle']['variance_min'],
                   'deltaq_variance_min_mean': endpoint[key]['predicted_endpoint_cycle'][
                       'variance_min_mean'],
                   'paper_discharge': paper_endpoint[key]['predicted_endpoint_cycle'][
                       'plus_capacity_all']}
            metrics = {}
            for name, predicted in eol.items():
                linear = coherent_curve(current, threshold, predicted-origin, horizons)
                shaped = power_curve(current, threshold, predicted-origin,
                                     horizons, power)
                metrics[name] = {
                    'eol_ape_percent': float(100*abs(predicted-cell.endpoint_cycle)/
                                             cell.endpoint_cycle),
                    'linear_curve_mape_percent': float(100*np.mean(abs(linear-true)/true)),
                    'power_curve_mape_percent': float(100*np.mean(abs(shaped-true)/true))}
            rows.append({'cell_id': key, 'group': group, 'origin_cycle': origin,
                         'future_points': len(future), 'metrics': metrics})
        folds.append({'held_out': group, 'fitted_power': power,
                      'training_event_cells': fitted, 'scored_cells': len(rows),
                      'by_cell': rows})
        print(group, 'n', len(rows), 'power', round(power, 2), flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    summary = {'scored_cells': len(rows)}
    for name in ('selected_direct', 'deltaq_variance_min',
                 'deltaq_variance_min_mean', 'paper_discharge'):
        summary[name] = {metric: float(np.mean([row['metrics'][name][metric]
                                               for row in rows]))
                         for metric in ('eol_ape_percent', 'linear_curve_mape_percent',
                                        'power_curve_mape_percent')}
    report = {'status': 'unpromoted MIT whole-batch diagnostic-assisted curve audit',
              'origin': 'cycle 100 (last observed at or before 100)',
              'input_warning': 'full Q(V) diagnostic at cycles 10 and 100',
              'endpoint_warning': 'source 0.88 Ah, not per-cell exact 80%',
              'summary': summary, 'folds': folds}
    (ROOT / 'deltaq_curve_audit.json').write_text(json.dumps(report, indent=2),
                                                 encoding='utf-8')
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
