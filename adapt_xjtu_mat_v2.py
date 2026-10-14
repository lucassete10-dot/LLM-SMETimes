"""Extract XJTU v2 inference signals from a protocol-matched original MAT.

Only the MAT ``data`` variable is loaded. In particular, ``summary`` capacity
and cycle-life fields are not read. The resulting NPZ contains exactly the
two arrays accepted by the frozen external-test signal stage.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from prepare_xjtu import OUT as TRAIN_PROCESSED, SOURCE as TRAIN_SOURCE
from prepare_xjtu import diagnostic, stage_features


def extract_from_cycles(cycles, batch):
    if batch not in range(1, 7):
        raise ValueError('XJTU batch must be 1 through 6')
    numbers, features = [], []
    for index, cycle in enumerate(cycles, 1):
        if not diagnostic(batch, cycle['description']):
            continue
        charge = stage_features(cycle, 0)
        discharge = stage_features(cycle, 2)
        if charge is None or discharge is None:
            continue
        numbers.append(index)
        features.append(np.r_[charge, discharge])
    if len(features) <= 5:
        raise ValueError('XJTU inference requires more than five complete diagnostics')
    cycle = np.asarray(numbers, dtype='int64')
    raw = np.asarray(features, dtype='float32')
    if raw.shape != (len(cycle), 46) or np.isinf(raw).any():
        raise ValueError('XJTU 46-column feature contract failed')
    return cycle, raw


def extract_mat(path, batch):
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != '.mat':
        raise ValueError(f'expected an existing MAT file: {path}')
    # Do not load the source summary: it holds target capacity and life labels.
    mat = loadmat(path, variable_names=['data'], simplify_cells=True)
    if 'summary' in mat or 'data' not in mat:
        raise ValueError('MAT variable selection failed')
    return extract_from_cycles(mat['data'], batch)


def verify_training_sources():
    checked = 0
    observations = 0
    for batch in range(1, 7):
        for path in sorted((TRAIN_SOURCE / f'Batch-{batch}').glob('*.mat')):
            cycle, raw = extract_mat(path, batch)
            with np.load(TRAIN_PROCESSED / f'b{batch}_{path.stem}.npz') as prepared:
                expected_cycle = prepared['cycle'].astype('int64')
                expected_raw = prepared['features'].astype('float32')
            if not np.array_equal(cycle, expected_cycle):
                raise ValueError(f'{path}: label-free cycle selection differs from training')
            if not np.array_equal(raw, expected_raw, equal_nan=True):
                raise ValueError(f'{path}: label-free features differ from training')
            checked += 1
            observations += len(cycle)
            print('MATCH', batch, path.stem, len(cycle), flush=True)
    if checked != 55:
        raise ValueError(f'expected 55 training cells, checked {checked}')
    print('VERIFIED', checked, 'XJTU cells and', observations, 'diagnostics', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    adapt = sub.add_parser('adapt', help='write label-free cycle/raw_features NPZ')
    adapt.add_argument('source_mat', type=Path)
    adapt.add_argument('batch', type=int)
    adapt.add_argument('output_npz', type=Path)
    sub.add_parser('verify-training', help='compare label-free extraction to all 55 training cells')
    args = parser.parse_args()
    if args.action == 'verify-training':
        verify_training_sources()
        return
    output = args.output_npz.resolve()
    if output.suffix.lower() != '.npz' or output.exists():
        raise ValueError('output must be a new .npz file; existing data is never overwritten')
    if not output.parent.is_dir():
        raise ValueError('output directory must already exist')
    cycle, raw = extract_mat(args.source_mat, args.batch)
    np.savez_compressed(output, cycle=cycle, raw_features=raw)
    print('SAVED', output, len(cycle), 'diagnostics; capacity and life not read', flush=True)


if __name__ == '__main__':
    main()
