"""Check shared voltage support in first 300 s of MIT cycles 10 and 100."""

import json
from pathlib import Path

import h5py
import numpy as np

from battery_v2_features import load_manifest
from prepare_mit import intervals, read_ref


OUT = Path('battery_models_v2/mit_qv_challenger/partial_source_audit.json')


def discharge_prefix(time, voltage, current, capacity, duration_s):
    """Select only samples through an observed discharge-prefix duration."""
    if duration_s <= 0:
        raise ValueError('duration must be positive')
    if not (len(time) == len(voltage) == len(current) == len(capacity)):
        return None
    within = None
    for run in intervals(current < -2):
        if len(run) < 3:
            continue
        elapsed = (time[run]-time[run[0]])*60
        if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) < 0):
            continue
        if np.any(elapsed >= duration_s):
            candidate = run[elapsed <= duration_s]
            prior = elapsed[np.searchsorted(elapsed, duration_s)-1]
            if len(candidate) >= 3 and prior >= duration_s-15:
                within = candidate
                break
    if within is None:
        return None
    v = voltage[within]
    q = capacity[within]
    if not (np.isfinite(v).all() and np.isfinite(q).all()):
        return None
    return v, q


def partial_trace(mat, index, requested, duration_s=300):
    struct = mat['batch']
    summary = mat[struct['summary'][index, 0]]
    cycles = mat[struct['cycles'][index, 0]]
    numbers = np.asarray(summary['cycle']).reshape(-1).astype(int)
    match = np.flatnonzero(numbers == requested)
    if len(match) != 1:
        return None
    j = int(match[0])
    voltage = read_ref(mat, cycles['V'][j, 0])
    current = read_ref(mat, cycles['I'][j, 0])
    time = read_ref(mat, cycles['t'][j, 0])
    qd = read_ref(mat, cycles['Qd'][j, 0])
    return discharge_prefix(time, voltage, current, qd, duration_s)


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    groups = {}
    for group in sorted({row['group'] for row in records}):
        subset = [row for row in records if row['group'] == group]
        rows = []
        with h5py.File(subset[0]['measurement_source'], 'r') as mat:
            for record in subset:
                index = int(record['cell_id'].split('c', 1)[1])
                early = partial_trace(mat, index, 10)
                later = partial_trace(mat, index, 100)
                if early is None or later is None:
                    rows.append({'cell_id': record['cell_id'], 'available': False})
                    continue
                lower = max(np.min(early[0]), np.min(later[0]))
                upper = min(np.max(early[0]), np.max(later[0]))
                rows.append({'cell_id': record['cell_id'], 'available': True,
                             'common_voltage_min': float(lower),
                             'common_voltage_max': float(upper),
                             'common_voltage_width': float(upper-lower)})
        valid = [row for row in rows if row['available']]
        groups[group] = {'cells': len(rows), 'paired_prefixes': len(valid),
                         'width_quantiles': np.quantile([row['common_voltage_width']
                                                         for row in valid], [0, .1, .5, .9, 1]).tolist(),
                         'lower_quantiles': np.quantile([row['common_voltage_min']
                                                         for row in valid], [0, .5, 1]).tolist(),
                         'upper_quantiles': np.quantile([row['common_voltage_max']
                                                         for row in valid], [0, .5, 1]).tolist(),
                         'by_cell': rows}
        print(group, {key: value for key, value in groups[group].items()
                      if key != 'by_cell'}, flush=True)
    report = {'purpose': 'shared cycle-10/100 Q(V) support in first 300 s',
              'online_note': 'Qd here is within-prefix current integration, not full-cycle capacity',
              'groups': groups}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
