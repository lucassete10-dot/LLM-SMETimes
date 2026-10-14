"""Label-free 120-second XJTU charge/discharge waveform adapter.

Loads MAT `data` only. `summary` contains capacity and cycle-life targets and
must never be loaded by this inference-side parser. Four channels per stage:
V, I, T, and prefix-integrated absolute throughput Q. Units: V, A, degC, Ah.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from prepare_xjtu import diagnostic


GRID_S = np.arange(0, 121, dtype=np.float64)
CHANNELS = ("voltage_V", "current_A", "temperature_C")


def stage_waveform(cycle, stage):
    time_min = np.asarray(cycle["relative_time_min"], dtype=np.float64).reshape(-1)
    starts = np.flatnonzero(time_min == 0)
    if len(starts) <= stage:
        return None
    start = starts[stage]
    stop = starts[stage + 1] if stage + 1 < len(starts) else len(time_min)
    time_s = 60 * time_min[start:stop]
    keep = np.isfinite(time_s) & (time_s >= 0) & (time_s <= GRID_S[-1])
    if keep.sum() < 90 or time_s[keep][-1] < GRID_S[-1] - 2:
        return None
    order = np.argsort(time_s[keep])
    unique, first = np.unique(time_s[keep][order], return_index=True)
    if len(unique) < 90:
        return None
    wave = []
    for channel in CHANNELS:
        source = np.asarray(cycle[channel], dtype=np.float64).reshape(-1)
        if len(source) != len(time_min):
            raise ValueError(f"{channel} length does not match time")
        values = source[start:stop][keep][order][first]
        if not np.isfinite(values).all():
            return None
        wave.append(np.interp(GRID_S, unique, values))
    voltage, current, temperature = wave
    if (np.any((voltage <= 0) | (voltage > 6)) or
            np.any(np.abs(current) > 20) or
            np.any((temperature < -40) | (temperature > 100))):
        return None
    throughput_ah = np.r_[0., np.cumsum((np.abs(current[1:]) +
                                        np.abs(current[:-1])) / 2 / 3600)]
    return np.stack((voltage, current, temperature, throughput_ah)).astype(np.float32)


def extract_cell_waveforms(cycles, batch):
    numbers, waveforms = [], []
    for index, cycle in enumerate(cycles, start=1):
        if not diagnostic(batch, cycle["description"]):
            continue
        charge = stage_waveform(cycle, 0)
        discharge = stage_waveform(cycle, 2)
        if charge is None or discharge is None:
            continue
        numbers.append(index)
        waveforms.append(np.stack((charge, discharge)))
    if not waveforms:
        raise ValueError("no valid charge/discharge diagnostic waveforms")
    return np.asarray(numbers, dtype=np.int64), np.stack(waveforms)


def extract_mat_waveforms(path, batch):
    path = Path(path)
    if path.suffix.lower() != ".mat" or not path.is_file():
        raise ValueError(f"expected original MAT file: {path}")
    source = loadmat(path, variable_names=["data"], simplify_cells=True)
    if "summary" in source or "data" not in source:
        raise ValueError("source unexpectedly loaded target summary")
    return extract_cell_waveforms(source["data"], batch)
