"""Evaluate a fixed late-knee SOH trajectory on saved held-out predictions.

The cubic shape is a predeclared challenger, not a fitted or promoted model.
It uses only the out-of-fold current SOH and RUL at the forecast origin.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from evaluate_anchored_trajectory_v2 import get_domain_arrays
from train_trajectory_v2 import threshold_soh


OUT = Path('battery_models_v2/knee_trajectory_challenger/development_cv.json')
POWER = 3.


def knee_curve(current_soh, threshold, rul, horizons, power=POWER):
    horizons = np.asarray(horizons, dtype='float64')
    if power < 1 or np.any(horizons < 0):
        raise ValueError('power must be >= 1 and horizons nonnegative')
    remaining = max(0., float(current_soh)-float(threshold))
    safe_rul = max(1., float(rul))
    curve = float(current_soh)-remaining*(horizons/safe_rul)**power
    return np.clip(curve, 0., float(current_soh))


def evaluate():
    _, cells = load_all()
    domains = {}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        soh_by_cell, rul_by_cell = get_domain_arrays(domain)
        rows = []
        for key, (rul_cycles, predicted_rul, _) in rul_by_cell.items():
            cell = cells[key]
            if cell.endpoint_cycle is None:
                continue
            origin = int(rul_cycles[0])
            soh_cycles, soh_predicted = soh_by_cell[key]
            pos = int(np.searchsorted(soh_cycles, origin))
            if pos >= len(soh_cycles) or soh_cycles[pos] != origin:
                raise ValueError(f'{key}: SOH origin missing')
            future = np.flatnonzero((cell.cycle > origin) &
                                    (cell.cycle <= cell.endpoint_cycle) &
                                    cell.soh_available)
            if len(future) < 2:
                continue
            current = float(soh_predicted[pos])
            curve = knee_curve(current, threshold_soh(cell), predicted_rul[0],
                               cell.cycle[future]-origin)
            truth = cell.soh[future]
            early = cell.cycle[future] <= .5*cell.endpoint_cycle
            late = cell.cycle[future] >= .75*cell.endpoint_cycle
            rows.append({'cell_id': key, 'group': cell.group,
                         'origin_cycle': origin, 'future_points': len(future),
                         'curve_mape_percent': float(100*np.mean(abs(curve-truth)/truth)),
                         'early_mape_percent': (float(100*np.mean(abs(curve[early]-truth[early])/
                                                                   truth[early])) if early.any() else None),
                         'late_mape_percent': (float(100*np.mean(abs(curve[late]-truth[late])/
                                                                  truth[late])) if late.any() else None),
                         'eol_ape_percent': float(100*abs(origin+predicted_rul[0]-
                                                            cell.endpoint_cycle)/cell.endpoint_cycle),
                         'physical_violations': int(np.count_nonzero(np.diff(curve) > 1e-9))})
        groups = sorted({r['group'] for r in rows})
        domains[domain] = {
            'scored_cells': len(rows),
            'equal_cell_curve_mape_percent': float(np.mean([r['curve_mape_percent'] for r in rows])),
            'equal_group_curve_mape_percent': float(np.mean([
                np.mean([r['curve_mape_percent'] for r in rows if r['group'] == group])
                for group in groups])),
            'early_mape_percent': float(np.mean([r['early_mape_percent'] for r in rows
                                                 if r['early_mape_percent'] is not None])),
            'late_mape_percent': float(np.mean([r['late_mape_percent'] for r in rows
                                                if r['late_mape_percent'] is not None])),
            'eol_mape_percent': float(np.mean([r['eol_ape_percent'] for r in rows])),
            'rows': rows}
        print(domain, 'n', len(rows), 'curve', round(domains[domain][
            'equal_cell_curve_mape_percent'], 2), 'EOL', round(domains[domain][
                'eol_mape_percent'], 2), flush=True)
    report = {'status': 'development held-out diagnostic; no promotion',
              'method': 'fixed cubic late-knee curve anchored to OOF SOH and RUL',
              'endpoint_warning': 'MIT and XJTU use source EOL, not exact per-cell 80% SOH',
              'domains': domains}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    evaluate()
