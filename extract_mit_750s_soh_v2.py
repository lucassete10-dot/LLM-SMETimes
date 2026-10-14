"""Extract current-cycle MIT 750 s discharge-prefix features without capacity.

Every source cycle is mapped to the previously accepted 300 s diagnostic row.
The feature uses only voltage, current, temperature, and time through 750 s;
source Qd/QDischarge and future cycles are never read as features.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from battery_v2_features import load_manifest
from prepare_mit import intervals, read_ref


OUT = Path('battery_models_v2/soh_long_prefix_challenger')
DURATION_S = 750
FEATURE_NAMES = ('v450', 'v600', 'v750', 'v750_minus_v300',
                 'v750_minus_v600', 'partial_q_at_3p08_v',
                 'partial_q_at_3p00_v', 'temperature_750_c',
                 'temperature_750_minus_300_c')


def prefix_features(time_min, current, voltage, temperature):
    """Return nine causal scalars from the first qualifying 750 s discharge."""
    arrays = [np.asarray(item, dtype=float).reshape(-1)
              for item in (time_min, current, voltage, temperature)]
    time_min, current, voltage, temperature = arrays
    if len({len(item) for item in arrays}) != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype='float32')
    for run in intervals(current < -2):
        if len(run) < 3:
            continue
        elapsed = (time_min[run]-time_min[run[0]])*60
        within = run[(elapsed >= 0) & (elapsed <= DURATION_S)]
        if len(within) < 3:
            continue
        t = (time_min[within]-time_min[run[0]])*60
        if (t[-1] < DURATION_S-15 or not np.isfinite(t).all() or
                np.any(np.diff(t) <= 0)):
            continue
        v, i, temp = voltage[within], current[within], temperature[within]
        if not (np.isfinite(v).all() and np.isfinite(i).all()):
            continue
        points = np.interp([300, 450, 600, 750], t, v)
        v300, v450, v600, v750 = points
        q = np.r_[0., np.cumsum(.5*(np.abs(i[1:])+np.abs(i[:-1]))*
                                 np.diff(t)/3600)]
        order = np.argsort(v)
        ordered_v, ordered_q = v[order], q[order]
        unique, first, count = np.unique(ordered_v, return_index=True,
                                         return_counts=True)
        average_q = np.add.reduceat(ordered_q, first)/count
        def q_at(voltage_v):
            return (float(np.interp(voltage_v, unique, average_q))
                    if unique[0] <= voltage_v <= unique[-1] else np.nan)
        t300 = float(np.interp(300, t, temp)) if np.isfinite(temp).all() else np.nan
        t750 = float(np.interp(750, t, temp)) if np.isfinite(temp).all() else np.nan
        return np.asarray((v450, v600, v750, v750-v300, v750-v600,
                           q_at(3.08), q_at(3.00), t750, t750-t300),
                          dtype='float32')
    return np.full(len(FEATURE_NAMES), np.nan, dtype='float32')


def extract_record(mat, record):
    index = int(record['cell_id'].split('c', 1)[1])
    struct = mat['batch']
    summary = mat[struct['summary'][index, 0]]
    source = mat[struct['cycles'][index, 0]]
    numbers = np.asarray(summary['cycle']).reshape(-1).astype(int)
    source_index = {number: j for j, number in enumerate(numbers)}
    with np.load(record['feature_source']) as prepared:
        accepted = prepared['cycle'].astype(int)
    result = []
    for number in accepted:
        j = source_index.get(int(number))
        if j is None:
            raise ValueError(f"{record['cell_id']}: missing source cycle {number}")
        result.append(prefix_features(
            read_ref(mat, source['t'][j, 0]),
            read_ref(mat, source['I'][j, 0]),
            read_ref(mat, source['V'][j, 0]),
            read_ref(mat, source['T'][j, 0])))
    return accepted, np.stack(result)


def main(max_cells=None):
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    if max_cells is not None:
        records = records[:max_cells]
    OUT.mkdir(parents=True, exist_ok=True)
    audit = []
    for group in sorted({row['group'] for row in records}):
        subset = [row for row in records if row['group'] == group]
        with h5py.File(subset[0]['measurement_source'], 'r') as mat:
            for row in subset:
                cycles, features = extract_record(mat, row)
                np.savez_compressed(OUT / f"{row['cell_id']}.npz",
                                    cycle=cycles, raw=features)
                observed = np.isfinite(features).any(axis=1)
                audit.append({
                    'cell_id': row['cell_id'], 'group': group,
                    'rows': len(cycles), '750s_available_rows': int(observed.sum()),
                    'finite_feature_fraction': float(np.isfinite(features).mean()),
                    'q_3p08_finite_fraction': float(np.isfinite(features[:, 5]).mean()),
                    'q_3p00_finite_fraction': float(np.isfinite(features[:, 6]).mean()),
                })
                print('EXTRACTED', row['cell_id'], 'rows', len(cycles),
                      '750 s available', int(observed.sum()), flush=True)
    report = {'status': 'causal MIT current-cycle 750 s voltage/current/temperature features',
              'duration_s': DURATION_S, 'feature_names': FEATURE_NAMES,
              'cells': len(audit), 'rows': sum(row['rows'] for row in audit),
              'available_rows': sum(row['750s_available_rows'] for row in audit),
              'by_cell': audit}
    (OUT / 'extraction_audit.json').write_text(json.dumps(report, indent=2),
                                               encoding='utf-8')
    print('SUMMARY', report['cells'], report['rows'], report['available_rows'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cells', type=int)
    main(parser.parse_args().max_cells)
