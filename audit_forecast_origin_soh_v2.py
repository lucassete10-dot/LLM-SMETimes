"""Audit present-SOH error at the first practical forecast origin per cell."""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from evaluate_anchored_trajectory_v2 import get_domain_arrays


OUT = Path('battery_models_v2/forecast_origin_soh_audit.json')


def evaluate():
    _, cells = load_all()
    domains = {}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        soh_by_cell, rul_by_cell = get_domain_arrays(domain)
        rows = []
        for key, (rul_cycle, _, _) in rul_by_cell.items():
            cell = cells[key]
            origin = int(rul_cycle[0])
            ci = int(np.searchsorted(cell.cycle, origin))
            if ci >= len(cell.cycle) or cell.cycle[ci] != origin:
                raise ValueError(f'{key}: missing cell origin')
            if not cell.soh_available[ci]:
                continue
            sc, se = soh_by_cell[key]
            si = int(np.searchsorted(sc, origin))
            if si >= len(sc) or sc[si] != origin:
                raise ValueError(f'{key}: missing SOH origin')
            actual = float(cell.soh[ci])
            predicted = float(se[si])
            rows.append({'cell_id': key, 'group': cell.group, 'origin_cycle': origin,
                         'true_soh': actual, 'estimated_soh': predicted,
                         'signed_error_soh': predicted-actual,
                         'ape_percent': 100*abs(predicted-actual)/actual})
        groups = sorted({r['group'] for r in rows})
        domains[domain] = {
            'scored_cells': len(rows),
            'first_origin_cell_macro_mape_percent': float(np.mean([r['ape_percent'] for r in rows])),
            'first_origin_equal_group_mape_percent': float(np.mean([
                np.mean([r['ape_percent'] for r in rows if r['group'] == group])
                for group in groups])),
            'worst_cells': sorted(rows, key=lambda r: r['ape_percent'], reverse=True)[:5],
            'by_group': {group: {'cells': sum(r['group'] == group for r in rows),
                                 'mape_percent': float(np.mean([r['ape_percent'] for r in rows
                                                                if r['group'] == group]))}
                         for group in groups}}
        print(domain, len(rows), round(domains[domain][
            'first_origin_cell_macro_mape_percent'], 2), flush=True)
    report = {'status': 'development held-out diagnostic',
              'interpretation': 'SOH error at the forecast origin is inherited by future curves',
              'domains': domains}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    evaluate()
