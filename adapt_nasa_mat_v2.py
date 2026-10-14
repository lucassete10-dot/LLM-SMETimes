"""Extract label-free NASA discharge-prefix features from an original MAT.

The MATLAB battery variable is loaded and discharge signal subfields are used;
the nested ``Capacity`` subfield is deliberately ignored. Unlike HDF5 MAT,
SciPy loads a whole MATLAB struct, so this means no capacity *access* rather
than selective exclusion from memory.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


GRID = np.arange(0, 1001, 100, dtype='float64')
PREFIX_SECONDS = 1000.


def as_vector(value):
    return np.asarray(value, dtype='float64').reshape(-1)


def prefix_features(data):
    time = as_vector(data['Time'])
    voltage = as_vector(data['Voltage_measured'])
    current = as_vector(data['Current_measured'])
    temperature = as_vector(data['Temperature_measured'])
    if not (len(time) == len(voltage) == len(current) == len(temperature)):
        raise ValueError('NASA discharge channels have inconsistent lengths')
    if len(time) < 2 or time[-1] < PREFIX_SECONDS or np.any(np.diff(time) < 0):
        raise ValueError('NASA discharge lacks a monotone 1000-second prefix')
    v_grid = np.interp(GRID, time, voltage)
    i_grid = np.interp(GRID, time, current)
    t_grid = np.interp(GRID, time, temperature)
    mask = time <= PREFIX_SECONDS
    prefix_time = np.concatenate([time[mask], [PREFIX_SECONDS]])
    prefix_voltage = np.concatenate([voltage[mask], [v_grid[-1]]])
    prefix_current = np.concatenate([current[mask], [i_grid[-1]]])
    prefix_temperature = np.concatenate([temperature[mask], [t_grid[-1]]])
    abs_current = np.abs(prefix_current)
    summary = np.asarray([
        v_grid[0] - v_grid[-1],
        prefix_voltage.mean(), prefix_voltage.std(),
        abs_current.mean(), prefix_current.std(),
        prefix_temperature.mean(), prefix_temperature.std(),
        t_grid[-1] - t_grid[0], prefix_temperature.max(),
        np.trapezoid(abs_current, prefix_time) / 3600.,
        np.trapezoid(abs_current * prefix_voltage, prefix_time) / 3600.,
    ], dtype='float64')
    return np.concatenate([v_grid, i_grid, t_grid, summary]).astype('float32')


def extract_records(records):
    rows = []
    for record in records:
        if record['type'] == 'discharge':
            rows.append(prefix_features(record['data']))
    if len(rows) <= 1:
        raise ValueError('NASA inference requires at least two complete discharges')
    raw = np.stack(rows)
    if raw.shape != (len(rows), 44) or np.isinf(raw).any():
        raise ValueError('NASA 44-column feature contract failed')
    return np.arange(1, len(rows) + 1, dtype='int64'), raw


def extract_mat(path, variable):
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != '.mat':
        raise ValueError(f'expected an existing NASA MAT: {path}')
    if not variable or not variable.isidentifier():
        raise ValueError('a MATLAB battery variable name is required')
    source = loadmat(path, variable_names=[variable], simplify_cells=True)
    if variable not in source or 'cycle' not in source[variable]:
        raise ValueError(f'{path}: missing battery variable/cycle structure')
    return extract_records(source[variable]['cycle'])


def verify_training():
    from battery_raw_mvp import load_raw_battery

    root = Path(r'D:\LLM\NASA\raw\extracted')
    total = 0
    for key in ('B0005', 'B0006', 'B0007', 'B0018'):
        cycle, raw = extract_mat(root / f'{key}.mat', key)
        trained = load_raw_battery(root, key, GRID, PREFIX_SECONDS)
        if not np.array_equal(cycle, trained.discharge_number):
            raise ValueError(f'{key}: discharge numbering mismatch')
        if not np.array_equal(raw, trained.features, equal_nan=True):
            raise ValueError(f'{key}: prefix feature mismatch')
        total += len(cycle)
        print('MATCH', key, len(cycle), flush=True)
    print('VERIFIED 4 NASA cells and', total, 'discharge prefixes', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    adapt = sub.add_parser('adapt')
    adapt.add_argument('source_mat', type=Path)
    adapt.add_argument('battery_variable')
    adapt.add_argument('output_npz', type=Path)
    sub.add_parser('verify-training')
    args = parser.parse_args()
    if args.action == 'verify-training':
        verify_training()
        return
    output = args.output_npz.resolve()
    if output.suffix.lower() != '.npz' or output.exists():
        raise ValueError('output must be a new .npz file; existing data is never overwritten')
    if not output.parent.is_dir():
        raise ValueError('output directory must already exist')
    cycle, raw = extract_mat(args.source_mat, args.battery_variable)
    np.savez_compressed(output, cycle=cycle, raw_features=raw)
    print('SAVED', output, len(cycle), 'discharges; Capacity subfield not accessed',
          flush=True)


if __name__ == '__main__':
    main()
