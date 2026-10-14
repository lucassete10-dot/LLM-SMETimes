"""Shared physical features from source-specific, capacity-free discharge prefixes.

Only quantities measurable by the current diagnostic and a previously
established per-cell reference are used. The window duration differs by
source and remains an explicit feature, rather than pretending equal samples.
"""

import warnings

import numpy as np


PREFIX_LAYOUT = {
    "NASA": (0, 11, 1000.0, 44),
    "CALCE": (0, 11, 1000.0, 29),
    "MIT": (0, 11, 300.0, 51),
    "XJTU": (23, 5, 120.0, 46),  # discharge follows charge's 23 columns
}

PHYSICAL_NAMES = (
    "voltage_start_v", "voltage_end_v", "voltage_mean_v",
    "voltage_std_v", "voltage_slope_mv_per_s", "current_abs_c",
    "current_std_c", "temperature_mean_c", "temperature_change_c",
    "prefix_discharge_fraction_of_reference",
)


def physical_prefix(raw, dataset_id, reference_capacity_ah):
    raw = np.asarray(raw, dtype=np.float64)
    start, count, duration, width = PREFIX_LAYOUT[dataset_id]
    if raw.ndim != 2 or raw.shape[1] != width:
        raise ValueError(f"{dataset_id}: expected [diagnostics, {width}] raw features")
    if not np.isfinite(reference_capacity_ah) or reference_capacity_ah <= 0:
        raise ValueError("positive established reference capacity required")
    v = raw[:, start:start + count]
    current = raw[:, start + count:start + 2 * count]
    temperature = (raw[:, start + 2 * count:start + 3 * count]
                   if dataset_id != "CALCE" else np.full_like(v, np.nan))
    if not np.isfinite(v).all() or not np.isfinite(current).all():
        raise ValueError(f"{dataset_id}: non-finite V/I prefix")
    grid = np.linspace(0, duration, count)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        temperature_mean = np.nanmean(temperature, axis=1)
    temperature_change = temperature[:, -1] - temperature[:, 0]
    throughput_fraction = (np.trapezoid(np.abs(current), grid, axis=1) /
                           3600 / reference_capacity_ah)
    physical = np.column_stack((
        v[:, 0], v[:, -1], v.mean(axis=1), v.std(axis=1),
        (v[:, -1] - v[:, 0]) * 1000 / duration,
        np.abs(current).mean(axis=1) / reference_capacity_ah,
        current.std(axis=1) / reference_capacity_ah,
        temperature_mean, temperature_change, throughput_fraction,
    ))
    if physical.shape[1] != len(PHYSICAL_NAMES):
        raise AssertionError("physical feature names mismatch")
    return physical.astype(np.float32), duration


def causal_shared_features(raw, dataset_id, reference_capacity_ah,
                           reference_measurement_count):
    """Raw physical levels + calibration deltas + causal five-point trends.

    The first reference diagnostics must be available before scoring any row.
    Calibration capacity is never copied into the feature matrix as a target.
    """
    physical, duration = physical_prefix(raw, dataset_id, reference_capacity_ah)
    if not 1 <= reference_measurement_count <= len(physical):
        raise ValueError("invalid calibration length")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        anchor = np.nanmedian(physical[:reference_measurement_count], axis=0)
    delta = physical - anchor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        trend = np.stack([delta[i] - np.nanmean(delta[max(0, i - 4):i + 1], axis=0)
                          for i in range(len(delta))])
    temperature_present = np.isfinite(physical[:, 7]).astype(np.float32)
    combined = np.column_stack((physical, delta, trend,
                                np.full(len(physical), duration / 1000),
                                temperature_present)).astype(np.float32)
    if np.isinf(combined).any():
        raise ValueError("infinite shared feature")
    return combined
