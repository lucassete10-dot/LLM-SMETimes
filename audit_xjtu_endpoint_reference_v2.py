"""Audit XJTU source life against two distinct 80% capacity definitions.

This does not change the SOH labels or train a model. The source MAT summary
contains one discharge-capacity value per original cycle, while the v2 label
contract retains only comparable full-discharge diagnostics. In particular,
the first raw cycle must not silently become the SOH reference: some groups
include partial discharges at the beginning of their records.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from battery_data_contract import confirmed_threshold_event


MANIFEST = Path('battery_data_v2/manifest.json')
OUT = Path('battery_models_v2/xjtu_endpoint_reference_audit.json')
NOMINAL_CAPACITY_AH = 2.0
NOMINAL_80_AH = 0.8 * NOMINAL_CAPACITY_AH


def first_crossing(cycle, capacity, threshold):
    indices = np.flatnonzero(capacity <= threshold)
    return int(cycle[indices[0]]) if len(indices) else None


def audit_cell(record):
    source = loadmat(record['measurement_source'], variable_names=['summary'],
                     simplify_cells=True)['summary']
    raw_capacity = np.asarray(source['discharge_capacity_Ah'], dtype=float).reshape(-1)
    source_life = int(np.asarray(source['cycle_life']).item())
    if len(raw_capacity) != source_life or record['endpoint_cycle'] != source_life:
        raise ValueError(f"{record['cell_id']}: source life does not align")
    with np.load(record['labels_file']) as labels:
        cycle = np.asarray(labels['cycle'], dtype=int)
        capacity = np.asarray(labels['capacity_ah'], dtype=float)
        eligible = np.asarray(labels['soh_available'], dtype=bool)
    if not np.allclose(capacity, raw_capacity[cycle - 1], atol=1e-5):
        raise ValueError(f"{record['cell_id']}: diagnostic capacity misalignment")
    cycle, capacity = cycle[eligible], capacity[eligible]
    own_80_ah = .8 * float(record['reference_capacity_ah'])
    return {
        'cell_id': record['cell_id'],
        'group': record['group'],
        'source_life_cycle': source_life,
        'source_first_cycle_capacity_ah': float(raw_capacity[0]),
        'source_last_cycle_capacity_ah': float(raw_capacity[-1]),
        'comparable_diagnostic_reference_ah': float(record['reference_capacity_ah']),
        'own_80_threshold_ah': own_80_ah,
        'nominal_80_threshold_ah': NOMINAL_80_AH,
        'last_eligible_diagnostic_cycle': int(cycle[-1]),
        'last_eligible_diagnostic_capacity_ah': float(capacity[-1]),
        'source_last_at_or_below_nominal_80': bool(raw_capacity[-1] <= NOMINAL_80_AH),
        'source_last_at_or_below_own_80': bool(raw_capacity[-1] <= own_80_ah),
        'first_own_80_diagnostic_cycle': first_crossing(cycle, capacity, own_80_ah),
        'confirmed_own_80_diagnostic_cycle': confirmed_threshold_event(
            cycle, capacity, own_80_ah),
        'first_nominal_80_diagnostic_cycle': first_crossing(
            cycle, capacity, NOMINAL_80_AH),
        'confirmed_nominal_80_diagnostic_cycle': confirmed_threshold_event(
            cycle, capacity, NOMINAL_80_AH),
    }


def summarize(rows):
    first = np.asarray([r['source_first_cycle_capacity_ah'] for r in rows])
    last = np.asarray([r['source_last_cycle_capacity_ah'] for r in rows])
    own = np.asarray([r['own_80_threshold_ah'] for r in rows])
    comparable_reference = np.asarray(
        [r['comparable_diagnostic_reference_ah'] for r in rows])
    comparable_at_source_life = [r for r in rows
                                 if r['last_eligible_diagnostic_cycle']
                                 == r['source_life_cycle']]
    return {
        'cells': len(rows),
        'source_first_capacity_ah_range': [float(first.min()), float(first.max())],
        'source_last_capacity_ah_range': [float(last.min()), float(last.max())],
        'source_life_has_eligible_diagnostic': len(comparable_at_source_life),
        'median_raw_source_last_capacity_to_comparable_reference_ratio': float(
            np.median(last / comparable_reference)),
        'median_comparable_source_life_soh': float(np.median([
            r['last_eligible_diagnostic_capacity_ah']
            / r['comparable_diagnostic_reference_ah']
            for r in comparable_at_source_life])) if comparable_at_source_life else None,
        'median_own_80_threshold_ah': float(np.median(own)),
        'median_nominal_minus_own_threshold_ah': float(np.median(NOMINAL_80_AH - own)),
        'source_last_at_or_below_nominal_80': sum(
            r['source_last_at_or_below_nominal_80'] for r in rows),
        'source_last_at_or_below_own_80': sum(
            r['source_last_at_or_below_own_80'] for r in rows),
        'first_own_80_diagnostic': sum(
            r['first_own_80_diagnostic_cycle'] is not None for r in rows),
        'confirmed_own_80_diagnostic': sum(
            r['confirmed_own_80_diagnostic_cycle'] is not None for r in rows),
        'first_nominal_80_diagnostic': sum(
            r['first_nominal_80_diagnostic_cycle'] is not None for r in rows),
        'confirmed_nominal_80_diagnostic': sum(
            r['confirmed_nominal_80_diagnostic_cycle'] is not None for r in rows),
    }


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    rows = [audit_cell(record) for record in manifest['cells']
            if record['dataset'] == 'XJTU']
    report = {
        'status': 'label-definition audit, not model validation',
        'source_endpoint': 'original MAT summary cycle_life',
        'own_80_reference': 'median first five comparable full-discharge diagnostics',
        'nominal_80_reference': '0.8 * 2.0 Ah; alternative, not selected SOH label',
        'confirmation_rule': 'three consecutive eligible diagnostics below threshold',
        'all': summarize(rows),
        'by_group': {group: summarize([r for r in rows if r['group'] == group])
                     for group in sorted({r['group'] for r in rows})},
        'by_cell': rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'all': report['all'], 'by_group': report['by_group']}, indent=2))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
