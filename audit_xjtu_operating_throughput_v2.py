"""Audit XJTU operating Ah as an optional, usage-aware prediction input.

Only MAT ``data`` is loaded: capacity/life fields in ``summary`` are never
used as features. Complete discharge-current traces from prior cycles *are*
used, so this is a different input regime from the 120/600-second-prefix
benchmarks; it must never be reported as capacity-free short-prefix inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


MANIFEST = Path('battery_data_v2/manifest.json')
OUT = Path('battery_models_v2/xjtu_operating_throughput_audit.json')
NOMINAL_AH = 2.0


def discharge_ah(entry):
    time = np.asarray(entry['relative_time_min'], dtype=float).reshape(-1)
    current = np.asarray(entry['current_A'], dtype=float).reshape(-1)
    if len(time) != len(current):
        raise ValueError('current/time arrays are not aligned')
    cuts = np.flatnonzero(time == 0)
    if len(cuts) <= 2:
        return np.nan
    start = int(cuts[2])
    stop = int(cuts[3]) if len(cuts) > 3 else len(time)
    t, i = time[start:stop], current[start:stop]
    good = np.isfinite(t) & np.isfinite(i)
    t, i = t[good], i[good]
    if len(t) < 2 or np.any(np.diff(t) < 0):
        return np.nan
    # Relative time is minutes. Negative current is discharge by source sign.
    return float(np.trapezoid(np.maximum(0, -i), t)/60)


def fill_missing_causally(values, nominal_ah=NOMINAL_AH):
    values = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(values)
    if np.any(np.isfinite(values) & ((values < 0) | (values > 3))):
        raise ValueError('nonphysical discharge-throughput value')
    for index in np.flatnonzero(missing):
        previous = values[max(0, index-20):index]
        previous = previous[np.isfinite(previous)]
        values[index] = float(np.median(previous)) if len(previous) else nominal_ah
    return values, int(missing.sum())


def audit_record(record):
    mat = loadmat(record['measurement_source'], variable_names=['data'],
                  simplify_cells=True)
    if 'summary' in mat or 'data' not in mat:
        raise ValueError('MAT variable selection failed')
    throughput, missing = fill_missing_causally(
        [discharge_ah(entry) for entry in mat['data']])
    with np.load(record['labels_file']) as labels:
        diagnostic_cycles = labels['cycle'].astype(int)
    origin_index = max(15, record['reference_measurement_count'])
    origin_cycle = (int(diagnostic_cycles[origin_index])
                    if origin_index < len(diagnostic_cycles) else None)
    endpoint = int(record['endpoint_cycle'])
    if len(throughput) < endpoint:
        raise ValueError(f"{record['cell_id']}: source cycle/endpoint mismatch")
    origin_admissible = origin_cycle is not None and origin_cycle < endpoint
    early = throughput[:origin_cycle-1] if origin_admissible else None
    return {'cell_id': record['cell_id'], 'group': record['group'],
            'origin_cycle': origin_cycle, 'endpoint_cycle': endpoint,
            'origin_admissible': origin_admissible,
            'imputed_source_cycles': missing,
            'early_mean_ah_per_cycle': float(np.mean(early)) if early is not None else None,
            'early_median_ah_per_cycle': float(np.median(early)) if early is not None else None,
            'early_equivalent_full_cycles': float(early.sum()/NOMINAL_AH)
            if early is not None else None,
            'endpoint_equivalent_full_cycles':
            float(throughput[:endpoint].sum()/NOMINAL_AH),
            'early_min_ah': float(early.min()) if early is not None else None,
            'early_max_ah': float(early.max()) if early is not None else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cells', type=int, default=None)
    args = parser.parse_args()
    records = [row for row in json.loads(MANIFEST.read_text(encoding='utf-8'))['cells']
               if row['dataset'] == 'XJTU']
    if args.max_cells is not None:
        records = records[:args.max_cells]
    rows = []
    for row in records:
        result = audit_record(row)
        rows.append(result)
        print(result['cell_id'], 'life', result['endpoint_cycle'],
              'early Ah/cycle', round(result['early_mean_ah_per_cycle'], 3)
              if result['origin_admissible'] else None,
              'lifetime EFC', round(result['endpoint_equivalent_full_cycles'], 1),
              flush=True)
    groups = sorted({row['group'] for row in rows})
    report = {'status': 'read-only usage audit; different input regime from prefix-only',
              'nominal_capacity_ah': NOMINAL_AH,
              'group_medians': {group: {
                  'cells': sum(row['group'] == group for row in rows),
                  'life_cycles': float(np.median([row['endpoint_cycle'] for row in rows
                                                  if row['group'] == group])),
                  'early_ah_per_cycle': float(np.median([
                      row['early_mean_ah_per_cycle'] for row in rows
                      if row['group'] == group and row['origin_admissible']])),
                  'lifetime_equivalent_full_cycles': float(np.median([
                      row['endpoint_equivalent_full_cycles'] for row in rows
                      if row['group'] == group]))}
                  for group in groups},
              'by_cell': rows}
    if args.max_cells is None:
        OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('WROTE', OUT)


if __name__ == '__main__':
    main()
