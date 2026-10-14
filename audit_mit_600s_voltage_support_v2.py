"""Label-free common voltage support for 600 s MIT cycle-10/100 prefixes."""

import json
import argparse
from pathlib import Path

import h5py
import numpy as np

from audit_mit_partial_qv_v2 import partial_trace
from battery_v2_features import load_manifest


OUT_ROOT = Path('battery_models_v2/mit_qv_challenger')
CANDIDATE_LOWER_V = (2.80, 2.85, 2.90, 2.95, 3.00, 3.02, 3.04,
                     3.06, 3.08, 3.10, 3.12, 3.15)
UPPER_V = 3.53


def main(duration_s=600, requested_cycles=(10, 100)):
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    rows = []
    for group in sorted({row['group'] for row in records}):
        subset = [row for row in records if row['group'] == group]
        with h5py.File(subset[0]['measurement_source'], 'r') as mat:
            for record in subset:
                index = int(record['cell_id'].split('c', 1)[1])
                traces = [partial_trace(mat, index, cycle, duration_s)
                          for cycle in requested_cycles]
                if any(trace is None for trace in traces):
                    rows.append({'cell_id': record['cell_id'], 'group': group,
                                 'available': False})
                    continue
                lower = max(float(np.min(trace[0])) for trace in traces)
                upper = min(float(np.max(trace[0])) for trace in traces)
                rows.append({'cell_id': record['cell_id'], 'group': group,
                             'available': True, 'common_lower_v': lower,
                             'common_upper_v': upper,
                             'samples_by_cycle': {
                                 str(number): len(trace[0]) for number, trace
                                 in zip(requested_cycles, traces)}})
    valid = [row for row in rows if row['available']]
    report = {
        'status': 'label-free discharge-prefix voltage-coverage audit',
        'duration_s': duration_s, 'cells': len(rows), 'paired_prefixes': len(valid),
        'requested_cycles': list(requested_cycles),
        'candidate_grid_coverage': {str(lower): {
            'all_cells': sum(row['common_lower_v'] <= lower and
                             row['common_upper_v'] >= UPPER_V for row in valid),
            'by_batch': {group: sum(row['group'] == group and
                                   row['common_lower_v'] <= lower and
                                   row['common_upper_v'] >= UPPER_V
                                   for row in valid)
                         for group in sorted({row['group'] for row in rows})},
        } for lower in CANDIDATE_LOWER_V},
        'common_lower_quantiles_v': np.quantile([
            row['common_lower_v'] for row in valid], [0, .1, .5, .9, 1]).tolist()
        if valid else None,
        'common_upper_quantiles_v': np.quantile([
            row['common_upper_v'] for row in valid], [0, .1, .5, .9, 1]).tolist()
        if valid else None,
        'by_cell': rows,
    }
    suffix = '' if tuple(requested_cycles) == (10, 100) else '_multi_cycle'
    out = OUT_ROOT / f'prefix_{duration_s}s_voltage_support{suffix}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('paired', len(valid), 'lower quantiles', report['common_lower_quantiles_v'])
    for lower, coverage in report['candidate_grid_coverage'].items():
        print('grid lower', lower, 'coverage', coverage, flush=True)
    print('WROTE', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, choices=(600, 750, 900),
                        default=600)
    parser.add_argument('--cycles', type=int, nargs='+', default=(10, 100))
    args = parser.parse_args()
    main(args.duration, tuple(args.cycles))
