"""Causal multi-scale signal view inspired by the supplied SOH papers.

This is a small-data adaptation, not a reproduction of their neural networks.
No capacity, measured SOH, endpoint, or future signal enters the features.
"""

import numpy as np


def time_gated_matrix(cell, fast_cycles=5.0, slow_cycles=50.0):
    """Return current/relative signal and two past-only, gap-aware scales."""
    cycle = np.asarray(cell.cycle, dtype=np.float64)
    raw = np.asarray(cell.raw, dtype=np.float32)
    delta = np.asarray(cell.delta, dtype=np.float32)
    if raw.ndim != 2 or raw.shape != delta.shape or len(cycle) != len(raw):
        raise ValueError("unaligned cycle and signal arrays")
    if not np.all(np.isfinite(cycle)) or np.any(np.diff(cycle) <= 0):
        raise ValueError("cycles must be finite and strictly increasing")
    if fast_cycles <= 0 or slow_cycles <= fast_cycles:
        raise ValueError("expected 0 < fast_cycles < slow_cycles")
    states = []
    for tau in (fast_cycles, slow_cycles):
        state = np.full(raw.shape[1], np.nan, dtype=np.float32)
        output = np.empty_like(delta)
        for index, observation in enumerate(delta):
            valid = np.isfinite(observation)
            if index:
                gate = np.exp(-(cycle[index] - cycle[index - 1]) / tau)
                previous = np.where(np.isfinite(state), state, observation)
                state = np.where(valid, gate * previous + (1 - gate) * observation, state)
            else:
                state = np.where(valid, observation, state)
            output[index] = state
        states.append(output)
    fast, slow = states
    observed_fraction = np.mean(np.isfinite(raw), axis=1, keepdims=True).astype(np.float32)
    return np.concatenate((raw, delta, fast, fast - slow, observed_fraction), axis=1)
