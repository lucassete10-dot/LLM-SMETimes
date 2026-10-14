"""Positive endpoint-anchored SOH trajectories and cellwise interval calibration.

The trajectory is a transparent coupling of a current-state estimate and a
remaining-life estimate, not an independent discovery of the endpoint.
"""

import numpy as np


def anchored_soh(current_soh, remaining_cycles, horizons, shape=1.0, threshold=.8):
    """Return a smooth monotone curve crossing threshold at remaining_cycles.

    A shifted power curve has a finite initial derivative for every positive
    shape. Beyond EOL a matched exponential tail preserves nonnegativity and
    the first derivative. A pre-EOL caller whose SOH estimate is below the
    threshold receives an explicit small upward projection of the anchor.
    """
    h = np.asarray(horizons, dtype=float)
    if (not np.isfinite(h).all() or np.any(h < 0) or
            not np.isfinite(current_soh) or not np.isfinite(remaining_cycles) or
            remaining_cycles <= 0 or not np.isfinite(shape) or shape <= 0 or
            not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("finite positive lifetime/shape and nonnegative horizons required")
    anchor = max(float(current_soh), threshold + 1e-4)
    x = h / remaining_cycles
    shift = .1
    denominator = (1 + shift) ** shape - shift ** shape
    progress = ((np.minimum(x, 1) + shift) ** shape - shift ** shape) / denominator
    before = anchor - (anchor - threshold) * progress
    terminal_rate = ((anchor - threshold) / remaining_cycles * shape *
                     (1 + shift) ** (shape - 1) / denominator)
    after = threshold * np.exp(-terminal_rate / threshold * np.maximum(h - remaining_cycles, 0))
    return np.where(h <= remaining_cycles, before, after)


def cellwise_log_rul_radius(calibration_rows, alpha=.1):
    """Split-conformal radius using one max-across-origins score per cell.

    Caller must supply residuals from cells absent from predictor fitting.
    Infinity is returned if the finite-sample rank exceeds calibration count.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    by_cell = {}
    for row in calibration_rows:
        actual, predicted = row["actual_rul"], row["predicted_rul"]
        if not np.isfinite(actual) or not np.isfinite(predicted) or actual < 0 or predicted < 0:
            raise ValueError("finite nonnegative RUL required")
        residual = abs(np.log1p(actual) - np.log1p(predicted))
        key = row["cell_id"]
        by_cell[key] = max(by_cell.get(key, 0), residual)
    if not by_cell:
        raise ValueError("empty calibration set")
    rank = int(np.ceil((len(by_cell) + 1) * (1 - alpha)))
    radius = float(sorted(by_cell.values())[rank - 1]) if rank <= len(by_cell) else float("inf")
    return radius, {"calibration_cells": len(by_cell), "rank": rank,
                    "alpha": alpha, "cell_scores": by_cell}


def log_rul_interval(remaining_cycles, radius):
    if not np.isfinite(remaining_cycles) or remaining_cycles < 0 or radius < 0 or np.isnan(radius):
        raise ValueError("nonnegative finite RUL and nonnegative radius required")
    center = np.log1p(remaining_cycles)
    return float(max(0, np.expm1(center - radius))), float(np.expm1(center + radius))
