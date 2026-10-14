"""Strictly clipped V/I samples from already-observed MIT/XJTU discharges.

No temperature, Qd, summary capacity or cycle-life fields are accessed. The
completed observed cycle is used for event segmentation; this is future-cycle
forecasting, not an intra-cycle real-time estimator.
"""

import hashlib
from contextlib import ExitStack
from pathlib import Path

import numpy as np


SCHEMA = "strict_raw_discharge_vi_120s_v2"
# MIT's late prefix cadence is about 5s. Allow one interval plus clock margin;
# the first v1 preparation trial's 2s gap wrongly rejected normal sampling.
# Chosen from source acquisition timestamps, never prediction/target errors.
MAXIMUM_ENDPOINT_GAP_SECONDS = 6.
VI_GRID = np.arange(0., 121., 30.)
VI_FEATURE_NAMES = ([f"voltage_change_since_0s_v_t{int(t)}s" for t in VI_GRID]
    + [f"absolute_current_over_rated_ah_c_t{int(t)}s" for t in VI_GRID]
    + ["voltage_at_0s_v"])
DIRECTIONS = (("MIT", "XJTU"), ("XJTU", "MIT"))
MINIMUM_OBSERVATIONS = {"MIT": 3, "XJTU": 6}


class RawVIReader:
    """Bounded-memory raw reader; never opens temperature/capacity fields."""

    def __init__(self, registry):
        self.registry = registry
        self.stack = ExitStack()
        self.handles = {}
        self.raw_hashes = {}
        self.xjtu_path, self.xjtu_records = None, None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.xjtu_records = None
        return self.stack.__exit__(*args)

    def extract(self, record, native, registered):
        domain = registered["metadata"]["dataset"]
        family = registered["physical_family"]
        specification = self.registry["families"][family]
        if domain not in MINIMUM_OBSERVATIONS or domain not in specification["datasets"]:
            raise ValueError("VI source dataset contradicts registered physical family")
        path = str(Path(record["raw_source"]).resolve())
        if path not in self.raw_hashes:
            self.raw_hashes[path] = stream_sha(path)
        previous = record.get("raw_source_sha256_from_existing_manifest")
        if previous and previous != self.raw_hashes[path]:
            raise ValueError("raw source differs from existing recorded fingerprint")
        capacity = specification["specifications"]["nominal_capacity_ah"]
        observed = np.asarray(native["cycle"])
        if observed.ndim != 1 or not np.isfinite(observed).all() or np.any(observed < 1) or np.any(observed != observed.astype(int)) or np.any(np.diff(observed) <= 0):
            raise ValueError("audited observed source cycles must be positive increasing integers")
        observed = observed[observed <= 100].astype(int).tolist()
        if domain == "MIT":
            import h5py
            if path not in self.handles:
                self.handles[path] = self.stack.enter_context(h5py.File(path, "r"))
            rows, rejected = extract_mit_vi(record["cell_id"], observed, self.handles[path], capacity)
        else:
            from scipy.io import loadmat
            if self.xjtu_path != path:
                self.xjtu_records = None  # do not retain 55 full raw files in RAM
                self.xjtu_records = loadmat(path, variable_names=["data"], simplify_cells=True)["data"]
                self.xjtu_path = path
            rows, rejected = extract_xjtu_vi(observed, self.xjtu_records, capacity)
        cycles = np.array([number for number, _ in rows], dtype=np.int64)
        values = np.array([value for _, value in rows], dtype=np.float32).reshape(-1, len(VI_FEATURE_NAMES))
        return cycles, values, rejected


def stream_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_vi_prefix(time_seconds, voltage, current, rated_capacity_ah, *, minimum_samples=3):
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples < 3:
        raise ValueError("minimum_samples must be an integer >=3")
    if isinstance(rated_capacity_ah, bool) or rated_capacity_ah is None or not np.isfinite(rated_capacity_ah) or rated_capacity_ah <= 0:
        raise ValueError("documented positive rated capacity required")
    t, v, i = [np.asarray(value, dtype=float).reshape(-1) for value in (time_seconds, voltage, current)]
    if not (len(t) == len(v) == len(i)):
        raise ValueError("inconsistent observed discharge channel lengths")
    take = np.isfinite(t) & (t >= 0) & (t <= 120)
    t, v, i = t[take], v[take], i[take]
    if len(t) < minimum_samples or t[0] > 2 or t[-1] < 120 - MAXIMUM_ENDPOINT_GAP_SECONDS:
        raise ValueError("incomplete strict 120-second prefix")
    if not np.isfinite(v).all() or not np.isfinite(i).all() or np.any(np.diff(t) < -1e-6):
        raise ValueError("nonfinite or nonmonotone early V/I prefix")
    t = np.maximum.accumulate(t)  # explicitly tolerated sub-microsecond roundoff
    t, index = np.unique(t, return_index=True)
    if len(t) < minimum_samples:
        raise ValueError("insufficient distinct early timestamps")
    # Interpolation never sees a point after 120 s. At 120, an endpoint recorded
    # at 114-120 s is held constant; no future sample is used to bracket it.
    vg, ig = np.interp(VI_GRID, t, v[index]), np.interp(VI_GRID, t, i[index])
    return np.r_[vg - vg[0], np.abs(ig) / rated_capacity_ah, vg[0]].astype(np.float32)


def extract_mit_vi(cell_id, observed_cycles, mat, rated_capacity_ah):
    from prepare_mit import discharge_slice, read_ref
    index = int(cell_id.split("c")[1])
    batch = mat["batch"]
    numbers = np.asarray(mat[batch["summary"][index, 0]]["cycle"]).reshape(-1).astype(int)
    positions = np.flatnonzero(numbers <= 100)
    mapping = {int(numbers[j]): int(j) for j in positions}
    if len(mapping) != len(positions) or set(observed_cycles) - set(mapping):
        raise ValueError("audited MIT cycles missing/duplicated in raw observed history")
    data = mat[batch["cycles"][index, 0]]
    rows, rejected = [], []
    for number in observed_cycles:
        if number < 1 or number > 100:
            raise ValueError("re-extraction is confined to observed source cycles 1-100")
        j = mapping[int(number)]
        t, v, i = [read_ref(mat, data[name][j, 0]) for name in ("t", "V", "I")]
        try:
            if not (len(t) == len(v) == len(i)):
                raise ValueError("inconsistent observed discharge channel lengths")
            run = discharge_slice(t, i, v)
            if run is None:
                raise ValueError("audited principal discharge event unavailable")
            seconds = (t[run] - t[run[0]]) * 60
            rows.append((int(number), strict_vi_prefix(seconds, v[run], i[run], rated_capacity_ah)))
        except ValueError as error:
            rejected.append({"cycle": int(number), "reason": str(error)})
    return rows, rejected


def extract_xjtu_vi(observed_cycles, records, rated_capacity_ah):
    rows, rejected = [], []
    for number in observed_cycles:
        if number < 1 or number > 100 or number > len(records):
            raise ValueError("audited XJTU cycle outside observed raw source history")
        cycle = records[int(number) - 1]
        try:
            time = np.asarray(cycle["relative_time_min"], dtype=float).reshape(-1)
            cuts = np.flatnonzero(time == 0)
            if len(cuts) <= 2:
                raise ValueError("discharge stage unavailable")
            start, stop = cuts[2], cuts[3] if len(cuts) > 3 else len(time)
            v, i = [np.asarray(cycle[name], dtype=float).reshape(-1) for name in ("voltage_V", "current_A")]
            if not (len(time) == len(v) == len(i)):
                raise ValueError("inconsistent observed discharge channel lengths")
            # The source uses stages charge/rest/discharge/rest; only stage2 V/I
            # and its local time are read. No charge or temperature fallback.
            rows.append((int(number), strict_vi_prefix(time[start:stop] * 60,
                v[start:stop], i[start:stop], rated_capacity_ah, minimum_samples=90)))
        except ValueError as error:
            rejected.append({"cycle": int(number), "reason": str(error)})
    return rows, rejected
