"""Read-only audit of MIT signal-only prefix extraction vs prepared features.

Reads cycle numbers from HDF5 summary metadata and I/V/t/T from cycle traces.
Never reads QDischarge, Qd, cycle-life, or prepared capacity/SOH arrays.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from prepare_mit import GRID, discharge_slice, feature_vector, read_ref


OUT = Path('battery_models_v2/mit_label_free_extraction_audit.json')
PROCESSED = Path(r'D:\LLM\MIT\processed')
RAW = Path(r'D:\LLM\MIT\raw')
MIT_SOURCE = {
    1: RAW / '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    2: RAW / '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
    3: RAW / '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}


def extract_cell(mat, batch, index):
    struct = mat['batch']
    summary = mat[struct['summary'][index, 0]]
    cycles = mat[struct['cycles'][index, 0]]
    cycle_number = np.asarray(summary['cycle']).reshape(-1).astype('int64')
    if len(cycle_number) != len(cycles['I']):
        raise ValueError(f'b{batch}c{index}: cycle number/trace mismatch')
    numbers, features = [], []
    rejected = {}
    for j, number in enumerate(cycle_number):
        try:
            current = read_ref(mat, cycles['I'][j, 0])
            voltage = read_ref(mat, cycles['V'][j, 0])
            time = read_ref(mat, cycles['t'][j, 0])
            temperature = read_ref(mat, cycles['T'][j, 0])
            if not (len(current) == len(voltage) == len(time) == len(temperature)):
                rejected[int(number)] = 'signal_length_mismatch'
                continue
            if not np.isfinite(np.c_[current, voltage, time]).all():
                rejected[int(number)] = 'invalid_signal'
                continue
            run = discharge_slice(time, current, voltage)
            if run is None:
                rejected[int(number)] = 'no_300s_main_discharge'
                continue
            raw = feature_vector(time[run], current[run], voltage[run],
                                 temperature[run], GRID)
            if len(raw) != 51 or np.isinf(raw).any():
                raise ValueError('invalid 51-column MIT signal feature')
            numbers.append(int(number))
            features.append(raw)
        except (KeyError, IndexError, ValueError, TypeError) as error:
            rejected[int(number)] = f'signal_error:{type(error).__name__}'
    if len(numbers) != len(set(numbers)) or np.any(np.diff(numbers) <= 0):
        raise ValueError(f'b{batch}c{index}: extracted cycle numbers not unique/increasing')
    return np.asarray(numbers, dtype='int64'), np.asarray(features, dtype='float32'), rejected


def main():
    groups = {}
    total_exact_features = 0
    for batch, source in MIT_SOURCE.items():
        folder = PROCESSED if batch == 1 else PROCESSED / f'batch{batch}'
        paths = sorted(folder.glob(f'b{batch}c*.npz'))
        if batch == 3:
            paths = [path for path in paths if path.stem != 'b3c37']
        cells = []
        with h5py.File(source, 'r') as mat:
            for path in paths:
                index = int(path.stem.split('c', 1)[1])
                cycle, raw, rejected = extract_cell(mat, batch, index)
                with np.load(path) as prepared:
                    expected_cycle = prepared['cycle'].astype('int64')
                    expected_raw = prepared['enhanced_raw'].astype('float32')
                lookup = {int(value): i for i, value in enumerate(cycle)}
                missing = sorted(set(expected_cycle.tolist()) - set(lookup))
                extra = sorted(set(cycle.tolist()) - set(expected_cycle.tolist()))
                if missing:
                    raise ValueError(f'{path.stem}: prepared signal rows absent from label-free extraction: {missing}')
                aligned = raw[[lookup[int(value)] for value in expected_cycle]]
                if not np.array_equal(aligned, expected_raw, equal_nan=True):
                    raise ValueError(f'{path.stem}: shared signal features differ')
                total_exact_features += len(expected_cycle)
                cells.append({'cell_id': path.stem,
                              'source_cycles': len(mat[mat['batch']['cycles'][index, 0]]['I']),
                              'signal_qualified': len(cycle), 'prepared': len(expected_cycle),
                              'extra_signal_only_cycle_numbers': extra,
                              'signal_rejected_count': len(rejected),
                              'signal_rejected_reason_counts': {reason: list(rejected.values()).count(reason)
                                                                 for reason in set(rejected.values())}})
                print('MATCH', path.stem, len(expected_cycle), 'extra', len(extra), flush=True)
        groups[f'batch{batch}'] = {
            'cells': len(cells), 'prepared_observations': sum(c['prepared'] for c in cells),
            'signal_qualified_observations': sum(c['signal_qualified'] for c in cells),
            'extra_signal_only_observations': sum(len(c['extra_signal_only_cycle_numbers'])
                                                  for c in cells),
            'cells_with_extras': sum(bool(c['extra_signal_only_cycle_numbers']) for c in cells),
            'cells_detail': cells}
        print('BATCH', batch, {key: value for key, value in groups[f'batch{batch}'].items()
                              if key != 'cells_detail'}, flush=True)
    report = {'purpose': 'MIT signal-only extraction audit; no capacity/life labels read',
              'exact_shared_feature_rows': total_exact_features, 'groups': groups}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
