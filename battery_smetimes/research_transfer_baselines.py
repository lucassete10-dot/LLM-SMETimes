"""Source-only baselines for the explicit shared-input research transfer task.

This module does not change production predictors or rename physical families.
Target labels are read only by evaluation, never by fitting or selection.
"""

import time
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from battery_smetimes.forecast import query_matrix
from battery_smetimes.research_baselines import regression_metrics
from battery_smetimes.research_transfer_signals import FEATURE_NAMES, RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import SCHEMA as VI_SCHEMA, VI_FEATURE_NAMES

CACHED_SCHEMA = "shared_discharge_cached_v1_not_native_smetimes"


def transfer_input_schema(cell):
    schema = cell.get("input_schema", CACHED_SCHEMA)
    names = {CACHED_SCHEMA: FEATURE_NAMES, VI_SCHEMA: VI_FEATURE_NAMES}.get(schema)
    if names is None or cell.get("feature_names", names) != names:
        raise ValueError("unknown or contradictory shared-input schema")
    return schema


def transfer_prefix(cell, origin):
    transfer_input_schema(cell)
    cycles, common = np.asarray(cell["cycle"]), np.asarray(cell["common"], dtype=float)
    if cycles.ndim != 1 or common.shape != (len(cycles), len(FEATURE_NAMES)):
        raise ValueError("shared-input schema mismatch; native predictors must not be used")
    if cell["reference_established_source_cycle"] > origin:
        raise ValueError("label reference not established at origin")
    take = cycles <= origin
    c, x = cycles[take], common[take]
    if len(c) < cell.get("minimum_signal_observations", 3):
        raise ValueError("insufficient shared visible history")
    if not np.isfinite(c).all() or np.any(np.diff(c) <= 0) or np.isinf(x).any() or not np.isfinite(x).any(axis=1).all():
        raise ValueError("invalid shared visible history")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        initial, last = x[0].copy(), np.nanmedian(x[-3:], axis=0)
        xx, tt = x[-min(10, len(x)):], c[-min(10, len(x)):].astype(float)
        valid = np.isfinite(xx)
        n = valid.sum(axis=0)
        t_mean = (valid * tt[:, None]).sum(axis=0) / np.maximum(n, 1)
        x_mean = np.nansum(xx, axis=0) / np.maximum(n, 1)
        centered = tt[:, None] - t_mean
        denominator = (centered * centered * valid).sum(axis=0)
        slope = np.divide(np.nansum(centered * (xx - x_mean), axis=0), denominator,
            out=np.full(x.shape[1], np.nan), where=(n >= 2) & (denominator > 0)) * 100
    return np.r_[last, initial, last - initial, slope, origin / 100,
                 c[-1] / 100, len(c) / 100, (c[-1] - c[0]) / 100]


def _allowed_cells(cells, plan):
    if plan["mode"] != RESEARCH_MODE or plan["target_calibration"] != []:
        raise ValueError("only explicit zero-target-label research transfer is implemented")
    source, target = set(plan["source_refit"]), set(plan["target_test"])
    if not source or not target or source & target or len(source) != len(plan["source_refit"]) or len(target) != len(plan["target_test"]):
        raise ValueError("transfer source/target IDs must be unique and disjoint")
    if plan["source_physical_family"] == plan["target_physical_family"]:
        raise ValueError("these transfer plans require distinct physical families")
    allowed = {}
    schemas = set()
    for keys, domain, family in ((source, plan["source_dataset"], plan["source_physical_family"]),
                                  (target, plan["target_dataset"], plan["target_physical_family"])):
        for key in keys:
            schemas.add(transfer_input_schema(cells[key]))
            if cells[key]["cell_id"] != key or cells[key]["metadata"]["dataset"] != domain or cells[key]["physical_family"] != family:
                raise ValueError("transfer provenance/family conflicts with the unchanged frozen plan")
            allowed[key] = {"dataset": domain, "physical_family": family}
    if len(schemas) != 1:
        raise ValueError("cannot mix V/I and V/I/T protocols despite equal feature width")
    return allowed


def training_pairs(cells, ids, origins):
    xs, ys, ws, used = [], [], [], []
    for key in ids:
        cell, parts = cells[key], []
        for origin in origins:
            try:
                prefix = transfer_prefix(cell, origin)
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
            used.append(key)
        for x, y in parts:
            xs.append(x)
            ys.append(y)
            ws.append(np.full(len(y), 1 / (len(parts) * len(y))))
    if not xs:
        raise ValueError("no valid source training future pairs")
    x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    weight *= len(weight) / weight.sum()
    return x, y, weight, used


def fit_transfer(cells, ids, plan, origins, model_name, parameters, seed, smoke=False):
    allowed = _allowed_cells(cells, plan)
    if not ids or len(ids) != len(set(ids)) or set(ids) - set(plan["source_refit"]):
        raise ValueError("target/duplicate/empty IDs cannot enter source-only model fitting")
    started = time.perf_counter()
    x, y, weight, used = training_pairs(cells, ids, origins)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imputer.fit_transform(x.astype(np.float64))
    scaler = None
    if model_name == "ridge":
        scaler = StandardScaler()
        offset = x[0].copy()
        scaler.fit(x - offset, sample_weight=weight)
        scaler.mean_ += offset
        x = scaler.transform(x)
        estimator = Ridge(alpha=parameters["alpha"], solver="svd")
    elif model_name == "extra_trees":
        estimator = ExtraTreesRegressor(n_estimators=48 if smoke else 160,
            min_samples_leaf=parameters["min_samples_leaf"], max_features=.8, random_state=seed, n_jobs=4)
    else:
        raise ValueError("this transfer executor currently supports Ridge and ExtraTrees only")
    estimator.fit(x, y, sample_weight=weight)
    return {"schema": "research_shared_transfer_baseline_v1", "mode": RESEARCH_MODE,
        "input_schema": transfer_input_schema(cells[ids[0]]),
        "plan": plan, "allowed_cells": allowed, "source_train_ids": list(ids), "used_source_train_ids": used,
        "source_only_preprocessor_fit_ids": list(ids), "imputer": imputer, "scaler": scaler,
        "model": estimator, "model_name": model_name, "parameters": dict(parameters), "seed": seed,
        "origins": list(origins), "training_pairs": len(y), "fit_seconds": time.perf_counter() - started,
        "smoke": smoke, "not_a_production_route": True}


def predict_transfer(bundle, cell, origin, future, *, mode):
    if mode != RESEARCH_MODE or bundle["mode"] != RESEARCH_MODE:
        raise ValueError("research-only mode required; cannot bypass production physical gate")
    if transfer_input_schema(cell) != bundle.get("input_schema", CACHED_SCHEMA):
        raise ValueError("query protocol differs from trained shared-input schema")
    identity = bundle["allowed_cells"].get(cell["cell_id"])
    if identity != {"dataset": cell["metadata"]["dataset"], "physical_family": cell["physical_family"]}:
        raise ValueError("query cell is not in the unchanged frozen transfer cohorts")
    if origin not in bundle["origins"]:
        raise ValueError("origin outside frozen experiment")
    x = query_matrix(transfer_prefix(cell, origin), origin, future)
    x = bundle["imputer"].transform(x.astype(float))
    if bundle["scaler"] is not None:
        x = bundle["scaler"].transform(x)
    prediction = bundle["model"].predict(x)
    if prediction.shape != (len(future),) or not np.isfinite(prediction).all():
        raise ValueError("invalid transfer predictions")
    return prediction


def evaluate_transfer(bundle, cells, ids, *, predictor=None):
    predictor = predictor or predict_transfer
    rows, skipped, arrays = [], [], {}
    for key in ids:
        cell = cells[key]
        for origin in bundle["origins"]:
            valid = np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin)
            future, truth = cell["label_cycle"][valid], cell["soh"][valid]
            if len(future) < 2:
                skipped.append({"cell_id": key, "origin": origin, "reason": "insufficient valid future targets"})
                continue
            try:
                prediction = predictor(bundle, cell, origin, future, mode=RESEARCH_MODE)
            except ValueError as error:
                if "insufficient" in str(error) or "not established" in str(error):
                    skipped.append({"cell_id": key, "origin": origin, "reason": str(error)})
                    continue
                raise
            tail = future >= (future[0] + future[-1]) / 2
            rows.append({"cell_id": key, "physical_family": cell["physical_family"], "origin": origin,
                "future_targets": len(future), "metrics": regression_metrics(truth, prediction),
                "within_horizon": {str(h): regression_metrics(truth[future <= origin + h], prediction[future <= origin + h])
                                   if np.any(future <= origin + h) else None for h in (10, 50, 100)},
                "late_half": regression_metrics(truth[tail], prediction[tail])})
            prefix = f"{key}__{origin}__"
            arrays.update({prefix + "cycle": future, prefix + "truth": truth, prefix + "prediction": prediction})
    summary = {str(origin): {"cells": len(subset),
        "cell_macro_mape_percent": float(np.mean([r["metrics"]["mape_percent"] for r in subset])) if subset else None,
        "fraction_cells_mape_le_6": float(np.mean([r["metrics"]["mape_percent"] <= 6 for r in subset])) if subset else None}
        for origin in bundle["origins"] for subset in ([r for r in rows if r["origin"] == origin],)}
    return {"summary": summary, "by_cell": rows, "skipped": skipped}, arrays
