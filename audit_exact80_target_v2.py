"""Audit observability of each cell's own 80%-SOH endpoint.

This is a label audit, not a predictive score. Cells without a confirmed
crossing are right-censored at their final available SOH observation.
"""

import json
from pathlib import Path

import numpy as np

from battery_data_contract import confirmed_threshold_event
from battery_v2_features import load_all


OUT = Path('battery_models_v2/exact80_target_audit.json')
TARGET_SOH = .8


def audit_cell(cell):
    observed = np.flatnonzero(cell.soh_available & np.isfinite(cell.soh))
    cycle = cell.cycle[observed]
    soh = cell.soh[observed]
    first = np.flatnonzero(soh <= TARGET_SOH)
    first_cycle = int(cycle[first[0]]) if len(first) else None
    confirmed = confirmed_threshold_event(cycle, soh, TARGET_SOH)
    source = cell.endpoint_cycle
    return {
        'cell_id': cell.cell_id,
        'group': cell.group,
        'reference_capacity_ah': float(cell.reference_capacity_ah),
        'exact80_threshold_ah': float(TARGET_SOH * cell.reference_capacity_ah),
        'source_endpoint_cycle': int(source) if source is not None else None,
        'first_observed_exact80_cycle': first_cycle,
        'first_confirmed_exact80_cycle': confirmed,
        'last_observed_soh_cycle': int(cycle[-1]) if len(cycle) else None,
        'source_minus_confirmed_cycles': int(source - confirmed)
        if source is not None and confirmed is not None else None,
    }


def summarize(rows):
    paired = [r['source_minus_confirmed_cycles'] for r in rows
              if r['source_minus_confirmed_cycles'] is not None]
    return {
        'cells': len(rows),
        'source_endpoint_cells': sum(r['source_endpoint_cycle'] is not None for r in rows),
        'first_crossing_cells': sum(r['first_observed_exact80_cycle'] is not None for r in rows),
        'confirmed_event_cells': sum(r['first_confirmed_exact80_cycle'] is not None for r in rows),
        'right_censored_for_confirmed_event': sum(r['first_confirmed_exact80_cycle'] is None
                                                   for r in rows),
        'source_exactly_matches_confirmed': sum(gap == 0 for gap in paired),
        'median_abs_source_confirmed_gap_cycles': float(np.median(np.abs(paired)))
        if paired else None,
    }


def main():
    manifest, cells = load_all()
    report = {
        'status': 'exact 80%-of-own-reference SOH label observability; not model validation',
        'soh_reference': 'per-cell median of initial calibration capacities',
        'confirmed_event': 'first three consecutive SOH <= 0.8 with local median confirmation',
        'domains': {},
    }
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        rows = [audit_cell(cells[r['cell_id']]) for r in manifest['cells']
                if r['dataset'] == domain]
        report['domains'][domain] = {
            **summarize(rows),
            'by_group': {group: summarize([r for r in rows if r['group'] == group])
                         for group in sorted({r['group'] for r in rows})},
            'by_cell': rows,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for domain, info in report['domains'].items():
        print(domain, 'cells', info['cells'], 'first crossing', info['first_crossing_cells'],
              'confirmed', info['confirmed_event_cells'], 'censored',
              info['right_censored_for_confirmed_event'], 'source exact matches',
              info['source_exactly_matches_confirmed'], flush=True)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
