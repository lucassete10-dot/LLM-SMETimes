"""Dataset-neutral battery labels and checks for SOH/RUL experiments.

This module owns label availability. Capacity measurements used to establish a
cell reference are calibration data, not blind SOH examples. RUL is supplied
only where an endpoint has actually been observed; censored cells retain their
last observed cycle for later survival-style modeling.
"""

from dataclasses import dataclass

import numpy as np


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class LabelSpec:
    reference_count: int
    reference_rule: str
    endpoint_rule: str
    endpoint_cycle: int | None = None
    endpoint_threshold_ah: float | None = None


def confirmed_threshold_event(cycle, capacity_ah, threshold_ah, consecutive=3,
                              confirmation_window=10):
    """First sustained crossing; isolated low capacity measurements are ignored.

    This is an explicitly derived endpoint, not an author's source cycle life.
    Full future measurements are allowed here because this constructs a target,
    never an input feature.
    """
    cycles = np.asarray(cycle)
    q = np.asarray(capacity_ah, dtype=float)
    for index in range(len(q) - consecutive + 1):
        if (np.all(q[index:index + consecutive] <= threshold_ah)
                and np.median(q[index:min(index + confirmation_window, len(q))])
                <= threshold_ah):
            return int(cycles[index])
    return None


def build_labels(cycle, capacity_ah, source_cycle, spec: LabelSpec):
    """Return homogeneous arrays and metadata without modifying source arrays."""
    cycles = np.asarray(cycle, dtype=np.int64).reshape(-1)
    capacities = np.asarray(capacity_ah, dtype=np.float64).reshape(-1)
    source = np.asarray(source_cycle, dtype=np.int64).reshape(-1)
    if not len(cycles) or not (len(cycles) == len(capacities) == len(source)):
        raise ValueError('cycle, capacity and source-cycle arrays must be nonempty and aligned')
    if np.any(np.diff(cycles) <= 0) or np.any(cycles < 1):
        raise ValueError('cycle numbers must be positive and strictly increasing')
    if np.any(~np.isfinite(capacities)) or np.any(capacities <= 0):
        raise ValueError('capacity must be finite and positive')
    if not 1 <= spec.reference_count < len(cycles):
        raise ValueError('reference_count must leave at least one prediction cycle')

    reference = float(np.median(capacities[:spec.reference_count]))
    reference_cycle = int(cycles[spec.reference_count - 1])
    soh = capacities / reference
    # A reference measured at the end of its own cycle cannot serve as a blind
    # prediction of that cycle's capacity. Start at the following observation.
    prediction_eligible = np.arange(len(cycles)) >= spec.reference_count
    soh[~prediction_eligible] = np.nan

    endpoint = spec.endpoint_cycle
    if endpoint is not None and endpoint < int(cycles[0]):
        raise ValueError('endpoint precedes the first observed cycle')
    rul = np.full(len(cycles), np.nan, dtype=np.float64)
    rul_eligible = np.zeros(len(cycles), dtype=bool)
    if endpoint is not None:
        rul_eligible = prediction_eligible & (cycles <= endpoint)
        rul[rul_eligible] = endpoint - cycles[rul_eligible]
    if not np.array_equal(np.isfinite(soh), prediction_eligible):
        raise AssertionError('SOH availability mismatch')
    if not np.array_equal(np.isfinite(rul), rul_eligible):
        raise AssertionError('RUL availability mismatch')

    # A review flag on the measured target, never an input feature or an
    # automatic deletion. A centered local median is permitted for an audit.
    local_median = np.array([np.median(capacities[max(0, i-2):min(len(cycles), i+3)])
                             for i in range(len(cycles))])
    capacity_review_flag = np.abs(capacities - local_median) > .04 * reference

    arrays = {
        'cycle': cycles, 'source_cycle': source, 'capacity_ah': capacities,
        'soh': soh, 'soh_available': prediction_eligible,
        'rul_cycles': rul, 'rul_available': rul_eligible,
        'capacity_review_flag': capacity_review_flag,
    }
    metadata = {
        'reference_capacity_ah': reference,
        'reference_established_at_cycle': reference_cycle,
        'reference_measurement_count': spec.reference_count,
        'reference_rule': spec.reference_rule,
        'endpoint_rule': spec.endpoint_rule,
        'endpoint_threshold_ah': spec.endpoint_threshold_ah,
        'endpoint_cycle': endpoint,
        'endpoint_observed': endpoint is not None,
        'censor_cycle': None if endpoint is not None else int(cycles[-1]),
        'observed_through_cycle': int(cycles[-1]),
        'soh_examples': int(prediction_eligible.sum()),
        'rul_examples': int(rul_eligible.sum()),
        'capacity_review_flags': int(capacity_review_flag.sum()),
        'post_endpoint_rows_excluded_from_rul': int((prediction_eligible & ~rul_eligible).sum())
            if endpoint is not None else 0,
    }
    return arrays, metadata
