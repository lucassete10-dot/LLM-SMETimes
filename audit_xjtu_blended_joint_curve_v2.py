"""Coherent held-out XJTU SOH curve from age-blended RUL endpoints.

The current SOH comes from the existing whole-strategy OOF fusion model. One
bounded linear curve crosses the source threshold at the predicted EOL; the
curve and RUL cannot report incompatible endpoints. This is a development
audit on the same folds used to motivate the blend, not an independent test.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_age_blend_v2 import (DIRECT, FRACTION, blend_endpoints,
                                         read_oof)
from train_trajectory_v2 import threshold_soh


SOH = Path('battery_models_v2/paper_inspired_soh_challenger/xjtu_development_oof_predictions.npz')
OUT = Path('battery_models_v2/xjtu_age_blend_challenger/joint_curve_audit.json')


def coherent_curve(current, threshold, remaining_cycles, horizons):
    current = float(current)
    delta = max(0., current-float(threshold))
    remaining = max(float(remaining_cycles), 1e-6)
    return np.clip(current-delta*np.asarray(horizons)/remaining, 0, current)


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'XJTU']
    cells = {row['cell_id']: load_cell(row) for row in records}
    direct = read_oof(DIRECT, 'estimate')
    fraction = read_oof(FRACTION, 'endpoint')
    with np.load(SOH) as source:
        soh = {(str(key), int(cycle)): float(estimate)
               for key, cycle, estimate in zip(source['cell_id'], source['cycle'],
                                               source['fusion'])}
    rows = []
    for key, cell in cells.items():
        cycles = sorted(cycle for cell_id, cycle in direct if cell_id == key)
        if not cycles or (key, cycles[0]) not in soh:
            continue
        origin = cycles[0]
        if origin >= cell.endpoint_cycle:
            continue
        future = np.flatnonzero((cell.cycle > origin) &
                                (cell.cycle <= cell.endpoint_cycle) &
                                cell.soh_available)
        if len(future) < 2:
            continue
        direct_eol = origin+direct[key, origin]
        fraction_eol = fraction[key, origin]
        blended_eol = float(blend_endpoints(
            np.array([origin]), np.array([direct[key, origin]]),
            np.array([fraction_eol]))[0])
        actual = cell.soh[future]
        horizons = cell.cycle[future]-origin
        predictions = {'direct': direct_eol, 'fraction': fraction_eol,
                       'blend': blended_eol}
        metrics = {}
        for name, endpoint in predictions.items():
            curve = coherent_curve(soh[key, origin], threshold_soh(cell),
                                   endpoint-origin, horizons)
            metrics[name] = {
                'curve_mape_percent': float(100*np.mean(abs(curve-actual)/actual)),
                'eol_ape_percent': float(100*abs(endpoint-cell.endpoint_cycle)/
                                          cell.endpoint_cycle),
                'predicted_endpoint_cycle': float(endpoint)}
        rows.append({'cell_id': key, 'group': cell.group, 'origin_cycle': origin,
                     'future_points': len(future), 'metrics': metrics})
    summary = {'scored_cells': len(rows)}
    for name in ('direct', 'fraction', 'blend'):
        summary[name] = {
            'curve_mape_percent': float(np.mean([
                row['metrics'][name]['curve_mape_percent'] for row in rows])),
            'eol_mape_percent': float(np.mean([
                row['metrics'][name]['eol_ape_percent'] for row in rows]))}
    report = {'status': 'unpromoted whole-strategy development joint-curve audit',
              'input': 'held-out fusion SOH and held-out source-EOL predictions',
              'curve': 'bounded linear crossing at predicted source EOL',
              'endpoint_warning': 'XJTU source EOL is not exact first per-cell 80% crossing',
              'summary': summary, 'by_cell': rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', summary)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
