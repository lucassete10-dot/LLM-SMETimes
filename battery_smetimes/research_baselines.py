"""Future-curve baselines with a hard physical-family gate and train-only scaling.

The early-signal extractor is reused, not rewritten. Its old provenance one-hot
is removed: batch/charging-direction is no longer a battery physical type.
"""

import time
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from battery_smetimes.forecast import prefix_vector, query_matrix, type_key
from battery_smetimes.physical_types import require_same_registered_family


def signal_prefix(cell, origin, conditions=True):
    if 'signal_schema' in cell['metadata']:
        from battery_smetimes.llm_external_schema import external_signal_prefix
        return external_signal_prefix(cell,origin,conditions)
    if cell["reference_established_source_cycle"] > origin:
        raise ValueError("label reference not established at origin")
    meta = cell["metadata"]
    old = prefix_vector(cell["cycle"], cell["raw"], meta, origin, [type_key(meta)])
    # Legacy tail = one old type indicator + five actual condition values.
    signals = old[:-6]
    return np.r_[signals, old[-5:]] if conditions else signals


def build_training_pairs(cells, ids, origins):
    x_parts, y_parts, weights = [], [], []
    used_ids = []
    for key in ids:
        cell = cells[key]
        parts = []
        for origin in origins:
            try:
                prefix = signal_prefix(cell, origin)
            except ValueError as error:
                if "insufficient" in str(error) or "not established" in str(error):
                    continue
                raise
            take = np.flatnonzero(np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin))
            if len(take) < 2:
                continue
            if len(take) > 96:
                take = take[np.unique(np.linspace(0, len(take) - 1, 96).round().astype(int))]
            parts.append((query_matrix(prefix, origin, cell["label_cycle"][take]), cell["soh"][take]))
        if parts:
            used_ids.append(key)
        for x, y in parts:
            x_parts.append(x)
            y_parts.append(y)
            weights.append(np.full(len(y), 1 / (len(parts) * len(y))))
    if not x_parts:
        raise ValueError("no valid training future pairs")
    x, y, w = np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(weights)
    # Equal contribution per battery, total scale equal to number of pairs.
    w *= len(w) / w.sum()
    return x, y, w, used_ids


def fit_baseline(cells, ids, family, registry, config, model_name, parameters, seed, smoke=False):
    if model_name not in ("ridge", "extra_trees"):
        from battery_smetimes.research_extended_baselines import fit_extended
        return fit_extended(cells, ids, family, registry, config, model_name, parameters, seed, smoke)
    require_same_registered_family(family, [cells[k]["physical_family"] for k in ids], registry)
    started = time.perf_counter()
    x, y, weight, used_ids = build_training_pairs(cells, ids, config["origins_source_cycles"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imputer.fit_transform(x.astype(np.float64))
    scaler = None
    if model_name == "ridge":
        scaler = StandardScaler()
        # Translate before weighted moment accumulation. Constant physical
        # conditions then become exactly zero, avoiding tiny negative roundoff
        # variances in sklearn's weighted update. Restore the original mean.
        offset = x[0].copy()
        scaler.fit(x - offset, sample_weight=weight)
        scaler.mean_ += offset
        x = scaler.transform(x)
        # Correlated sampled voltages/trends are expected, especially XJTU.
        # SVD in float64 avoids float32 Cholesky ill-conditioning; not a
        # test-score-driven change to regularization or the training cohort.
        estimator = Ridge(alpha=parameters["alpha"], solver="svd")
    elif model_name == "extra_trees":
        estimator = ExtraTreesRegressor(n_estimators=48 if smoke else 160,
            min_samples_leaf=parameters["min_samples_leaf"], max_features=.8,
            random_state=seed, n_jobs=4)
    else:
        raise ValueError("unknown baseline")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        estimator.fit(x, y, sample_weight=weight)
    return {"schema": "soh_research_v5_baseline", "model": estimator,
        "model_name": model_name, "parameters": dict(parameters), "physical_family": family,
        "registry": registry, "imputer": imputer, "scaler": scaler, "seed": seed,
        "train_ids": list(ids), "used_train_ids": used_ids, "training_pairs": len(y),
        "input_contract": "early V/I/T and condition statistics; no legacy type/batch one-hot; no measured Q/SOH/EOL",
        "origins": config["origins_source_cycles"], "fit_seconds": time.perf_counter() - started,
        "smoke": smoke}


def predict_baseline(bundle, cell, origin, future_cycles):
    require_same_registered_family(cell["physical_family"], [bundle["physical_family"]], bundle["registry"])
    x = query_matrix(signal_prefix(cell, origin), origin, future_cycles)
    if bundle["model_name"] == "physical_family_mean":
        templates = bundle["templates"]
        if any(np.any(np.diff(row["cycle"]) <= 0) for row in templates):
            raise ValueError("training mean curves have nonmonotone source cycles")
        return np.mean([np.interp(future_cycles, row["cycle"], row["soh"]) for row in templates], axis=0)
    if "feature_columns" in bundle:
        x = x[:, bundle["feature_columns"]]
    x = bundle["imputer"].transform(x.astype(np.float64))
    if bundle["scaler"] is not None:
        x = bundle["scaler"].transform(x)
    prediction = bundle["model"].predict(x)
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite prediction")
    return prediction


def regression_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if not len(truth) or not np.isfinite(truth).all() or np.any(truth <= 0):
        raise ValueError("invalid evaluation targets")
    residual = prediction - truth
    total = float(np.sum((truth - truth.mean()) ** 2))
    return {"rmse_soh": float(np.sqrt(np.mean(residual ** 2))),
        "mae_soh": float(np.mean(np.abs(residual))),
        "mape_percent": float(100 * np.mean(np.abs(residual) / truth)),
        "r2": None if total <= 1e-12 else float(1 - np.sum(residual ** 2) / total)}


def evaluate_baseline(bundle, cells, ids, predictor=None):
    predictor = predictor or predict_baseline
    rows, skipped, arrays = [], [], {}
    for key in ids:
        cell = cells[key]
        for origin in bundle["origins"]:
            take = np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin)
            future, truth = cell["label_cycle"][take], cell["soh"][take]
            if len(future) < 2:
                skipped.append({"cell_id": key, "origin": origin, "reason": "insufficient valid future targets"})
                continue
            started = time.perf_counter()
            try:
                prediction = predictor(bundle, cell, origin, future)
            except ValueError as error:
                if "insufficient" in str(error) or "not established" in str(error):
                    skipped.append({"cell_id": key, "origin": origin, "reason": str(error)})
                    continue
                raise
            milliseconds = (time.perf_counter() - started) * 1000
            row = {"cell_id": key, "physical_family": cell["physical_family"], "origin": origin,
                "future_targets": len(truth), "metrics": regression_metrics(truth, prediction),
                "whole_curve_inference_ms": milliseconds,
                "within_horizon": {str(h): regression_metrics(truth[future <= origin + h], prediction[future <= origin + h])
                                   if np.any(future <= origin + h) else None for h in (10, 50, 100)}}
            tail = future >= (future[0] + future[-1]) / 2
            row["late_half"] = regression_metrics(truth[tail], prediction[tail])
            rows.append(row)
            prefix = f"{key}__{origin}__"
            arrays.update({prefix + "cycle": future, prefix + "truth": truth, prefix + "prediction": prediction})
    summary = {}
    for origin in bundle["origins"]:
        subset = [row for row in rows if row["origin"] == origin]
        summary[str(origin)] = {"cells": len(subset),
            "cell_macro_mape_percent": float(np.mean([row["metrics"]["mape_percent"] for row in subset])) if subset else None,
            "fraction_cells_mape_le_6": float(np.mean([row["metrics"]["mape_percent"] <= 6 for row in subset])) if subset else None}
    return {"summary": summary, "by_cell": rows, "skipped": skipped}, arrays
