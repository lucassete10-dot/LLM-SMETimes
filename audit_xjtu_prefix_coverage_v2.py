"""Read-only check of longer XJTU diagnostic-prefix availability.

Uses only MAT ``data`` and the already accepted source-cycle indices. This
does not create new features, read target capacities, or modify the candidate.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


MANIFEST = Path('battery_data_v2/manifest.json')
OUT = Path('battery_models_v2/xjtu_prefix_coverage_audit.json')
WINDOWS_S = (120, 300, 600)


def stage_time(cycle, stage):
    relative = np.asarray(cycle['relative_time_min'], dtype='float64').reshape(-1)
    cuts = np.flatnonzero(relative == 0)
    if len(cuts) <= stage:
        return np.empty(0, dtype='float64')
    begin = cuts[stage]
    end = cuts[stage + 1] if stage + 1 < len(cuts) else len(relative)
    return 60 * relative[begin:end]


def audit():
    records = json.loads(MANIFEST.read_text(encoding='utf-8'))['cells']
    groups = {}
    cells = []
    for record in records:
        if record['dataset'] != 'XJTU':
            continue
        with np.load(record['labels_file']) as labels:
            accepted = labels['cycle'].astype('int64')
        source = loadmat(record['measurement_source'], variable_names=['data'],
                         simplify_cells=True)
        if 'data' not in source or 'summary' in source:
            raise ValueError(f"{record['cell_id']}: data-only load failed")
        source_cycles = source['data']
        counts = {str(window): 0 for window in WINDOWS_S}
        durations = []
        for number in accepted:
            cycle = source_cycles[int(number) - 1]
            charge = stage_time(cycle, 0)
            discharge = stage_time(cycle, 2)
            if not len(charge) or not len(discharge):
                raise ValueError(f"{record['cell_id']}: missing stage at source cycle {number}")
            durations.append((float(np.nanmax(charge)), float(np.nanmax(discharge))))
            for window in WINDOWS_S:
                if (np.isfinite(charge).sum() >= 90 and np.isfinite(discharge).sum() >= 90
                        and np.nanmax(charge) >= window - 2
                        and np.nanmax(discharge) >= window - 2):
                    counts[str(window)] += 1
        row = {
            'cell_id': record['cell_id'], 'group': record['group'],
            'accepted_diagnostics': len(accepted),
            'both_stage_duration_coverage': counts,
            'median_charge_stage_duration_s': float(np.median([x[0] for x in durations])),
            'median_discharge_stage_duration_s': float(np.median([x[1] for x in durations])),
        }
        cells.append(row)
    for group in sorted({row['group'] for row in cells}):
        subset = [row for row in cells if row['group'] == group]
        total = sum(row['accepted_diagnostics'] for row in subset)
        groups[group] = {
            'cells': len(subset), 'accepted_diagnostics': total,
            'both_stage_duration_coverage': {
                key: sum(row['both_stage_duration_coverage'][key] for row in subset)
                for key in map(str, WINDOWS_S)},
            'both_stage_duration_fraction': {
                key: sum(row['both_stage_duration_coverage'][key] for row in subset) / total
                for key in map(str, WINDOWS_S)},
        }
    return {'status': 'read-only duration coverage; signal values and labels not inspected',
            'groups': groups, 'cells': cells}


if __name__ == '__main__':
    result = audit()
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', OUT)
    for group, row in result['groups'].items():
        print(group, row['accepted_diagnostics'], row['both_stage_duration_fraction'], flush=True)
