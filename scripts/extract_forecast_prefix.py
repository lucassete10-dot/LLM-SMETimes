"""Convert one raw MIT/XJTU/NASA cell into a label-free forecast input."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from adapt_nasa_mat_v2 import prefix_features
from prepare_mit import GRID, discharge_slice, feature_vector, read_ref
from prepare_xjtu import diagnostic, stage_features
from scripts.audit_mit_discharge_integrity import discontinuity


def extract(path, metadata, origin, *, cell_index=None, variable=None):
    rows = []
    if metadata["dataset"] == "MIT":
        if cell_index is None or cell_index < 0:
            raise ValueError("MIT requires zero-based cell-index in its batch MAT")
        with h5py.File(path, "r") as mat:
            batch = mat["batch"]
            numbers = np.asarray(mat[batch["summary"][cell_index, 0]]["cycle"]).reshape(-1).astype(int)
            data = mat[batch["cycles"][cell_index, 0]]
            for j in np.flatnonzero(numbers <= origin):
                t, v, i, temp = [read_ref(mat, data[name][j, 0]) for name in ("t", "V", "I", "T")]
                if not (len(t) == len(v) == len(i) == len(temp)) or not np.isfinite(np.c_[t, v, i]).all():
                    continue
                if discontinuity(t, i)[0] is not None:
                    continue
                run = discharge_slice(t, i, v)
                if run is None:
                    continue
                try:
                    feature = feature_vector(t[run], i[run], v[run], temp[run], GRID)
                except ValueError:
                    continue
                if not np.isinf(feature).any():
                    rows.append((int(numbers[j]), feature))
    elif metadata["dataset"] == "XJTU":
        batch = int(metadata["group"].replace("batch", ""))
        data = loadmat(path, variable_names=["data"], simplify_cells=True)["data"]
        for number, record in enumerate(data[:origin], 1):
            if diagnostic(batch, record["description"]):
                charge, discharge = stage_features(record, 0), stage_features(record, 2)
                if charge is not None and discharge is not None:
                    rows.append((number, np.r_[charge, discharge]))
    elif metadata["dataset"] == "NASA":
        if not variable:
            raise ValueError("NASA requires MATLAB battery variable, e.g. B0018")
        data = loadmat(path, variable_names=[variable], simplify_cells=True)[variable]["cycle"]
        number = 0
        for record in data:
            if record["type"] != "discharge":
                continue
            number += 1
            if number > origin:
                break
            try:
                feature = prefix_features(record["data"])
                if np.isfinite(feature).all():
                    rows.append((number, feature))
            except ValueError:
                continue
    else:
        raise ValueError("unsupported dataset")
    if not rows:
        raise ValueError("no valid signal prefixes")
    return np.asarray([r[0] for r in rows], dtype=np.int64), np.asarray([r[1] for r in rows], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--origin", type=int, default=100)
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--variable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir() or args.output.suffix != ".npz":
        raise ValueError("output must be a new .npz in an existing folder")
    if args.origin <= 0:
        raise ValueError("origin must be positive")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    cycle, raw = extract(args.source, metadata, args.origin, cell_index=args.cell_index, variable=args.variable)
    np.savez_compressed(args.output, cycle=cycle, raw_features=raw)
    print("SAVED", args.output, len(cycle), "signal observations; no capacity/SOH/life labels accessed")


if __name__ == "__main__":
    main()
