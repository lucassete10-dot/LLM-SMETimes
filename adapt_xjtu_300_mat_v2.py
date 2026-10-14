"""Signal-only XJTU 300-second MAT adapter for the development challenger.

The source ``summary`` variable containing capacity/life is never loaded.
This schema is separate from the frozen 120-second XJTU input contract.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from extract_xjtu_long_prefix_v2 import stage_features
from prepare_xjtu import diagnostic


def extract_mat(path, batch):
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != '.mat':
        raise ValueError(f'expected an existing XJTU MAT: {path}')
    if batch not in range(1, 7):
        raise ValueError('batch must be 1 through 6')
    source = loadmat(path, variable_names=['data'], simplify_cells=True)
    if 'summary' in source or 'data' not in source:
        raise ValueError('MAT data-only variable selection failed')
    cycle, rows = [], []
    for number, entry in enumerate(source['data'], 1):
        if not diagnostic(batch, entry['description']):
            continue
        charge = stage_features(entry, 0, 300)
        discharge = stage_features(entry, 2, 300)
        if charge is None or discharge is None:
            continue
        cycle.append(number)
        rows.append(np.r_[charge, discharge])
    raw = np.asarray(rows, dtype='float32')
    cycle = np.asarray(cycle, dtype='int64')
    if raw.shape != (len(cycle), 82) or len(cycle) <= 5 or np.isinf(raw).any():
        raise ValueError('300s XJTU signal extraction failed')
    return cycle, raw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_mat', type=Path)
    parser.add_argument('batch', type=int)
    parser.add_argument('output_npz', type=Path)
    args = parser.parse_args()
    output = args.output_npz.resolve()
    if output.suffix.lower() != '.npz' or output.exists():
        raise ValueError('output must be a new .npz; existing data is never overwritten')
    if not output.parent.is_dir():
        raise ValueError('output directory must already exist')
    cycle, raw = extract_mat(args.source_mat, args.batch)
    np.savez_compressed(output, cycle=cycle, raw_features=raw)
    print('SAVED', output, len(cycle), '300s diagnostics; capacity/life not read')
