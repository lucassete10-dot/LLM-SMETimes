"""Extract signal-only MIT 4C discharge prefixes from Severson-format MAT.

The extractor reads the summary's cycle-number metadata and the I/V/t/T trace
references. It never reads QDischarge, Qd, SOH, or cycle-life targets. A
signal-qualified new cycle is kept even if old training preparation would
have excluded it using a target-capacity screen; see the audit report.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from audit_mit_label_free_extraction_v2 import extract_cell


def extract_mat(path, batch, cell_index):
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != '.mat':
        raise ValueError(f'expected an existing MIT HDF5 MAT: {path}')
    if batch not in (1, 2, 3) or cell_index < 0:
        raise ValueError('batch must be 1/2/3 and cell index nonnegative')
    with h5py.File(path, 'r') as mat:
        if 'batch' not in mat or cell_index >= len(mat['batch']['cycles']):
            raise ValueError('MAT does not contain the requested batch/cell structure')
        cycle, raw, _ = extract_cell(mat, batch, cell_index)
    if len(cycle) <= 1 or raw.shape != (len(cycle), 51):
        raise ValueError('MIT inference requires at least two complete 51-column prefixes')
    return cycle, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_mat', type=Path)
    parser.add_argument('batch', type=int, choices=(1, 2, 3))
    parser.add_argument('cell_index', type=int)
    parser.add_argument('output_npz', type=Path)
    args = parser.parse_args()
    output = args.output_npz.resolve()
    if output.suffix.lower() != '.npz' or output.exists():
        raise ValueError('output must be a new .npz file; existing data is never overwritten')
    if not output.parent.is_dir():
        raise ValueError('output directory must already exist')
    cycle, raw = extract_mat(args.source_mat, args.batch, args.cell_index)
    np.savez_compressed(output, cycle=cycle, raw_features=raw)
    print('SAVED', output, len(cycle), 'signal-qualified cycles; capacity/life not read',
          flush=True)


if __name__ == '__main__':
    main()
