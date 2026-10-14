"""Coherent MIT future curve from whole-batch OOF SOH and EOL predictions.

The power shape is fitted on observed-endpoint batteries from the two outer
training batches, then applied to the third. MIT's 0.88 Ah source endpoint is
not exact per-cell 80% SOH; censored cells cannot receive EOL error scores.
"""

import json
from pathlib import Path

import numpy as np

from audit_xjtu_blended_joint_curve_v2 import coherent_curve
from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_age_blend_v2 import blend_endpoints, read_oof
from evaluate_xjtu_trained_knee_v2 import fit_power, power_curve
from train_soh_v2 import outer_splits
from train_trajectory_v2 import threshold_soh


SOH = Path('battery_models_v2/paper_inspired_soh_challenger/mit_development_oof_predictions.npz')
DIRECT = Path('battery_models_v2/rul/mit/development_oof_predictions.npz')
FRACTION = Path('battery_models_v2/mit_fractional_life_challenger/development_oof_endpoint.npz')
OUT = Path('battery_models_v2/mit_fractional_life_challenger/joint_curve_audit.json')


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    direct = read_oof(DIRECT, 'estimate')
    fraction = read_oof(FRACTION, 'fractional_endpoint')
    with np.load(SOH) as source:
        soh = {(str(key), int(cycle)): float(value)
               for key, cycle, value in zip(source['cell_id'], source['cycle'],
                                             source['gated'])}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        training_event_cells = [key for key in train
                                if cells[key].endpoint_cycle is not None]
        power, fitted = fit_power(cells, training_event_cells)
        rows = []
        for key in validation:
            cell = cells[key]
            if cell.endpoint_cycle is None:
                continue
            origin = max(15, cell.reference_count)
            if origin >= len(cell.cycle):
                continue
            cycle = int(cell.cycle[origin])
            if cycle >= cell.endpoint_cycle or (key, cycle) not in soh:
                continue
            future = np.flatnonzero((cell.cycle > cycle) &
                                    (cell.cycle <= cell.endpoint_cycle) &
                                    cell.soh_available)
            if len(future) < 2:
                continue
            current = soh[key, cycle]
            threshold = threshold_soh(cell)
            direct_eol = cycle+direct[key, cycle]
            fractional_eol = fraction[key, cycle]
            blended_eol = float(blend_endpoints(np.array([cycle]),
                                 np.array([direct[key, cycle]]),
                                 np.array([fractional_eol]))[0])
            horizon = cell.cycle[future]-cycle
            true = cell.soh[future]
            metrics = {}
            for name, endpoint in (('direct', direct_eol),
                                   ('fraction', fractional_eol),
                                   ('blend', blended_eol)):
                linear = coherent_curve(current, threshold, endpoint-cycle, horizon)
                shaped = power_curve(current, threshold, endpoint-cycle,
                                     horizon, power)
                metrics[name] = {
                    'linear_curve_mape_percent': float(100*np.mean(
                        abs(linear-true)/true)),
                    'power_curve_mape_percent': float(100*np.mean(
                        abs(shaped-true)/true)),
                    'eol_ape_percent': float(100*abs(
                        endpoint-cell.endpoint_cycle)/cell.endpoint_cycle)}
            oracle = power_curve(current, threshold, cell.endpoint_cycle-cycle,
                                 horizon, power)
            rows.append({'cell_id': key, 'group': group, 'origin_cycle': cycle,
                         'future_points': len(future), 'metrics': metrics,
                         'endpoint_oracle_power_curve_mape_percent': float(100*np.mean(
                             abs(oracle-true)/true))})
        folds.append({'held_out': group, 'fitted_power': power,
                      'training_endpoint_cells': fitted, 'scored_cells': len(rows),
                      'by_cell': rows})
        print(group, 'p', round(power, 2), 'n', len(rows), 'linear',
              round(float(np.mean([r['metrics']['blend']['linear_curve_mape_percent']
                                   for r in rows])), 2), 'power',
              round(float(np.mean([r['metrics']['blend']['power_curve_mape_percent']
                                   for r in rows])), 2), flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    summary = {'scored_cells': len(rows)}
    for name in ('direct', 'fraction', 'blend'):
        summary[name] = {metric: float(np.mean([
            row['metrics'][name][metric] for row in rows]))
            for metric in ('linear_curve_mape_percent', 'power_curve_mape_percent',
                           'eol_ape_percent')}
    summary['endpoint_oracle_power_curve_mape_percent'] = float(np.mean([
        row['endpoint_oracle_power_curve_mape_percent'] for row in rows]))
    report = {'status': 'unpromoted whole-batch MIT development curve audit',
              'input': 'held-out capacity-free SOH and EOL estimates',
              'shape': 'power fitted on outer-training batches; linear after predicted crossing',
              'endpoint': 'source 0.88 Ah; 12 right-censored cells unscored',
              'summary': summary, 'folds': folds}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
