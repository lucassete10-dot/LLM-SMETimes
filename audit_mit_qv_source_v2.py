"""Read-only check of MIT cycles 10/100 discharge Q(V) availability.

This diagnostic uses full observed within-cycle discharge-capacity traces and
is not a capacity-free online feature. It reads no future-cycle measurements.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from battery_v2_features import load_manifest
from prepare_mit import discharge_slice, read_ref


OUT = Path('battery_models_v2/mit_qv_challenger')
CYCLES = (10, 100)


def cycle_trace(mat, index, requested):
    struct = mat['batch']
    summary = mat[struct['summary'][index, 0]]
    cycles = mat[struct['cycles'][index, 0]]
    cycle_number = np.asarray(summary['cycle']).reshape(-1).astype(int)
    match = np.flatnonzero(cycle_number == requested)
    if len(match) != 1:
        return None, 'missing_or_ambiguous_cycle'
    j = int(match[0])
    voltage = read_ref(mat, cycles['V'][j, 0])
    current = read_ref(mat, cycles['I'][j, 0])
    time = read_ref(mat, cycles['t'][j, 0])
    capacity = read_ref(mat, cycles['Qd'][j, 0])
    if not (len(voltage) == len(current) == len(time) == len(capacity)):
        return None, 'length_mismatch'
    if not np.isfinite(np.c_[voltage, current, time, capacity]).all():
        return None, 'nonfinite_trace'
    run = discharge_slice(time, current, voltage)
    if run is None:
        return None, 'no_main_discharge'
    return (voltage[run], capacity[run]), None


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    groups = {}
    for group in sorted({row['group'] for row in records}):
        subset = [row for row in records if row['group'] == group]
        source = Path(subset[0]['measurement_source'])
        rows = []
        with h5py.File(source, 'r') as mat:
            for record in subset:
                index = int(record['cell_id'].split('c', 1)[1])
                row = {'cell_id': record['cell_id'], 'cycles': {}}
                for requested in CYCLES:
                    trace, reason = cycle_trace(mat, index, requested)
                    if trace is None:
                        row['cycles'][str(requested)] = {'reason': reason}
                    else:
                        voltage, capacity = trace
                        row['cycles'][str(requested)] = {
                            'points': len(voltage),
                            'voltage_min': float(np.min(voltage)),
                            'voltage_max': float(np.max(voltage)),
                            'capacity_min': float(np.min(capacity)),
                            'capacity_max': float(np.max(capacity))}
                rows.append(row)
        paired = [row for row in rows if all('voltage_min' in row['cycles'][str(c)]
                                             for c in CYCLES)]
        lower = np.asarray([max(row['cycles']['10']['voltage_min'],
                                row['cycles']['100']['voltage_min']) for row in paired])
        upper = np.asarray([min(row['cycles']['10']['voltage_max'],
                                row['cycles']['100']['voltage_max']) for row in paired])
        groups[group] = {'cells': len(rows), 'paired_traces': len(paired),
                         'cover_2p5_to_3p4': int(np.sum((lower <= 2.5) & (upper >= 3.4))),
                         'lower_common_voltage_quantiles': np.quantile(lower, [0, .5, 1]).tolist(),
                         'upper_common_voltage_quantiles': np.quantile(upper, [0, .5, 1]).tolist(),
                         'by_cell': rows}
        print(group, {key: value for key, value in groups[group].items()
                      if key != 'by_cell'}, flush=True)
    report = {'purpose': 'MIT paper-inspired full-discharge Q(V) source audit',
              'measurement': 'observed cycle 10 and 100 main 4C discharge Qd/V traces',
              'online_warning': 'requires complete discharge traces; not 300 s prefix-only',
              'groups': groups}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'source_audit.json').write_text(json.dumps(report, indent=2),
                                           encoding='utf-8')
    print('WROTE', OUT / 'source_audit.json', flush=True)


if __name__ == '__main__':
    main()
