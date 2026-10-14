"""Shared label/future-removal prediction checks; no training or model loading."""

import copy
import time
import numpy as np
from battery_smetimes.research_transfer_signals import RESEARCH_MODE


def prefix_only(cell, origin):
    visible = copy.deepcopy(cell)
    for key in ("soh", "label_cycle", "unfiltered_soh"):
        visible.pop(key, None)
    mask = np.asarray(visible["cycle"]) <= origin
    visible["cycle"] = visible["cycle"][mask].copy()
    visible["common"] = visible["common"][mask].copy()
    return visible


def causal_predictions(predictor, bundle, original, visible, poisoned, origin, future, intact):
    estimator = bundle.get("model")
    forest = estimator is not None and hasattr(estimator, "estimators_") and hasattr(estimator, "n_jobs")
    if not forest:
        return (intact, predictor(bundle, visible, origin, future, mode=RESEARCH_MODE),
            predictor(bundle, poisoned, origin, future, mode=RESEARCH_MODE), {"serial_tree_accumulation_control": False})
    jobs = estimator.n_jobs
    try:
        estimator.n_jobs = 1
        values = tuple(predictor(bundle, cell, origin, future, mode=RESEARCH_MODE) for cell in (original, visible, poisoned))
    finally:
        estimator.n_jobs = jobs
    return (*values, {"serial_tree_accumulation_control": True, "original_n_jobs": jobs, "control_n_jobs": 1,
        "original_runtime_restored": True,
        "scope": "paired causal check only; original runtime retained for saved-prediction matching"})


def replay_predictions(bundle, cells, plan, saved_arrays, *, predictor, tolerance,
                       origins=None, max_cells=None, progress_callback=None):
    """All selected target points; exact paired causality, fixed save tolerance."""
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid fixed replay tolerance")
    selected = list(bundle["origins"] if origins is None else origins)
    if not selected or len(selected) != len(set(selected)) or set(selected)-set(bundle["origins"]):
        raise ValueError("empty/duplicate/out-of-contract shared replay origins")
    if max_cells is not None and (isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells < 1):
        raise ValueError("max_cells must be a positive integer or all cells")
    ids = plan["target_test"] if max_cells is None else plan["target_test"][:max_cells]
    rows, outputs = [], {}
    for key in ids:
        original = cells[key]
        for origin in selected:
            prefix = f"{key}__{origin}__"
            if prefix+"prediction" not in saved_arrays:
                rows.append({"cell_id": key, "origin": origin, "status": "not originally eligible; retained as coverage gap"})
                if progress_callback:
                    progress_callback(rows, outputs)
                continue
            future, saved = np.asarray(saved_arrays[prefix+"cycle"]), np.asarray(saved_arrays[prefix+"prediction"])
            if (future.ndim != 1 or future.size < 2 or saved.shape != future.shape or not np.isfinite(future).all()
                    or np.any(future <= origin) or not np.isfinite(saved).all()):
                raise ValueError("invalid saved shared target grid/prediction")
            started = time.perf_counter()
            intact = predictor(bundle, original, origin, future, mode=RESEARCH_MODE)
            visible = prefix_only(original, origin)
            poisoned = copy.deepcopy(original)
            for name in ("soh", "label_cycle", "unfiltered_soh"):
                poisoned.pop(name, None)
            poisoned["common"][poisoned["cycle"] > origin] = np.inf
            poisoned["cycle"] = np.concatenate([poisoned["cycle"], [max(origin, float(np.max(original["cycle"])))+1]])
            poisoned["common"] = np.concatenate([poisoned["common"], np.full((1, original["common"].shape[1]), np.inf)])
            causal_intact, truncated, perturbed, control = causal_predictions(predictor, bundle, original, visible, poisoned, origin, future, intact)
            if any(value.shape != saved.shape or not np.isfinite(value).all() for value in (intact, causal_intact, truncated, perturbed)):
                raise ValueError("shared replay predictor returned invalid shape or nonfinite values")
            match = bool(np.allclose(intact, saved, rtol=tolerance, atol=tolerance))
            causal = bool(np.array_equal(causal_intact, truncated) and np.array_equal(causal_intact, perturbed))
            row = {"cell_id": key, "origin": origin, "future_points": len(future), "saved_prediction_match": match,
                "same_device_no_future_invariance": causal, "max_absolute_soh_difference": float(np.max(np.abs(intact-saved))),
                "atol": tolerance, "rtol": tolerance,
                "causal_max_absolute_difference": float(max(np.max(np.abs(causal_intact-truncated)), np.max(np.abs(causal_intact-perturbed)))),
                "causal_control": control, "labels_removed": True, "future_input_rows_deleted": True,
                "poisoned_future_row_appended": True, "seconds": time.perf_counter()-started, "passed": match and causal}
            rows.append(row)
            outputs.update({prefix+"cycle": future, prefix+"intact": intact, prefix+"causal_intact": causal_intact,
                prefix+"prefix_only": truncated, prefix+"poisoned_future": perturbed, prefix+"saved_prediction": saved})
            if progress_callback:
                progress_callback(rows, outputs)
            if not row["passed"]:
                raise ValueError("shared checkpoint failed saved-prediction or causal replay; do not loosen tolerances")
    if not any(row.get("passed") for row in rows):
        raise ValueError("shared replay selected no originally evaluated observations")
    return rows, outputs
