"""Consistent held-out SOH and RUL first-origin full-curve scorecard.

Each domain uses its selected development SOH/RUL route. A bounded linear
curve is anchored to the selected predicted threshold-crossing cycle, so
curve MAPE and EOL error refer to the same deployable pair of inputs.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from evaluate_anchored_trajectory_v2 import get_domain_arrays
from train_trajectory_v2 import threshold_soh


ROOT = Path('battery_models_v2')
OUT = ROOT / 'selected_joint_curve_all_audit.json'
SOH = {
    'NASA': (ROOT / 'nasa_boundary_soh_challenger/development_oof_predictions.npz', 'estimate'),
    'CALCE': (ROOT / 'paper_inspired_soh_challenger/calce_development_oof_predictions.npz', 'fusion'),
    'MIT': (ROOT / 'paper_inspired_soh_challenger/mit_development_oof_predictions.npz', 'gated'),
    'XJTU': (ROOT / 'paper_inspired_soh_challenger/xjtu_development_oof_predictions.npz', 'fusion'),
}
RUL = {
    'NASA': None,
    'CALCE': (ROOT / 'paper_inspired_soh_challenger/calce_rul_development_oof.npz', 'gated_raw'),
    'MIT': None,
    'XJTU': (ROOT / 'paper_inspired_soh_challenger/xjtu_rul_development_oof.npz', 'gated_log'),
}
SOH_ROUTE = {'NASA': 'boundary_extrapolation', 'CALCE': 'fusion',
             'MIT': 'time_gated', 'XJTU': 'fusion'}


def read_selected(path, field, cells):
    with np.load(path) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    result = {}
    for key in cells:
        mask = data['cell_id'] == key
        if mask.any():
            result[key] = (data['cycle'][mask], data[field][mask])
    return result


def score_domain(domain, cells, soh, rul):
    rows = []
    for key, (cycles, predicted_rul) in rul.items():
        cell = cells[key]
        if cell.endpoint_cycle is None or len(cycles) == 0:
            continue
        origin = int(cycles[0])
        sc, values = soh[key]
        j = int(np.searchsorted(sc, origin))
        if j >= len(sc) or sc[j] != origin:
            raise ValueError(f'{key}: selected SOH/RUL cycle mismatch')
        origin_soh = float(values[j])
        future = np.flatnonzero((cell.cycle > origin) &
                                (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
        if len(future) < 2:
            continue
        threshold = threshold_soh(cell)
        remaining = max(0., origin_soh-threshold)
        h = cell.cycle[future]-origin
        curve = np.clip(origin_soh-remaining*h/max(1., float(predicted_rul[0])),
                        0., origin_soh)
        truth = cell.soh[future]
        actual_at_origin = int(np.searchsorted(cell.cycle, origin))
        present_ape = (float(100*abs(origin_soh-cell.soh[actual_at_origin])/
                             cell.soh[actual_at_origin])
                       if cell.soh_available[actual_at_origin] else None)
        rows.append({'cell_id': key, 'group': cell.group,
                     'origin_cycle': origin, 'future_points': len(future),
                     'origin_ape_percent': present_ape,
                     'curve_mape_percent': float(100*np.mean(abs(curve-truth)/truth)),
                     'eol_ape_percent': float(100*abs(origin+predicted_rul[0]-
                                                      cell.endpoint_cycle)/cell.endpoint_cycle),
                     'origin_below_threshold': bool(origin_soh <= threshold)})
    groups = sorted({r['group'] for r in rows})
    summary = {'scored_cells': len(rows),
               'cell_macro_curve_mape_percent': float(np.mean([
                   r['curve_mape_percent'] for r in rows])),
               'equal_group_curve_mape_percent': float(np.mean([
                   np.mean([r['curve_mape_percent'] for r in rows if r['group'] == group])
                   for group in groups])),
               'worst_cell_curve_mape_percent': float(max([
                   r['curve_mape_percent'] for r in rows])),
               'cell_macro_eol_mape_percent': float(np.mean([
                   r['eol_ape_percent'] for r in rows])),
               'origin_soh_mape_percent': float(np.mean([
                   r['origin_ape_percent'] for r in rows
                   if r['origin_ape_percent'] is not None])),
               'origin_below_threshold_cells': sum(r['origin_below_threshold'] for r in rows)}
    print(domain, summary, flush=True)
    return {'soh_route': SOH_ROUTE[domain],
            'rul_route': RUL[domain][1] if RUL[domain] else 'baseline',
            'summary': summary, 'by_cell': rows}


def main():
    _, all_cells = load_all()
    report = {'status': 'development whole-cell/batch held-out audit, not independent test',
              'curve': 'bounded linear, crosses source EOL threshold at predicted RUL',
              'endpoint_warning': 'MIT and XJTU use source-defined endpoint, not exact per-cell 80%',
              'domains': {}}
    for domain in SOH:
        cells = {key: cell for key, cell in all_cells.items() if cell.dataset == domain}
        soh = read_selected(*SOH[domain], cells)
        _, baseline_rul = get_domain_arrays(domain)
        if RUL[domain]:
            rul = read_selected(*RUL[domain], cells)
        else:
            rul = {key: (cycle, estimate) for key, (cycle, estimate, _) in
                   baseline_rul.items()}
        if set(soh) != set(cells) or set(rul) != set(baseline_rul):
            raise ValueError(f'{domain}: selected OOF coverage mismatch')
        for key in rul:
            np.testing.assert_array_equal(rul[key][0], baseline_rul[key][0])
        report['domains'][domain] = score_domain(domain, cells, soh, rul)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', OUT, flush=True)


if __name__ == '__main__':
    main()
