"""Small supervised target-domain SOH offset with cell-balanced fitting."""

import numpy as np


def fit_cell_balanced_offset(calibration):
    """Median of per-cell mean residuals; no evaluation-cell data accepted.

    `calibration` maps target cell IDs to `(base_prediction, measured_soh)`.
    Each cell receives equal influence irrespective of diagnostic density.
    """
    if not calibration:
        raise ValueError("at least one calibration cell is required")
    residuals = []
    for cell_id, pair in calibration.items():
        prediction, target = (np.asarray(value, dtype=np.float64)
                              for value in pair)
        if (prediction.ndim != 1 or prediction.shape != target.shape or
                prediction.size == 0 or not np.isfinite(prediction).all() or
                not np.isfinite(target).all() or np.any(target <= 0)):
            raise ValueError(f"invalid calibration labels: {cell_id}")
        residuals.append(float(np.mean(target - prediction)))
    return float(np.median(residuals))


def apply_offset(prediction, offset):
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    return np.asarray(prediction, dtype=np.float64) + offset
