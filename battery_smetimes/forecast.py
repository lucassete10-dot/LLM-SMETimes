"""Future SOH from a label-free signal prefix and known operating conditions."""

import re
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

ORIGINS = (20, 50, 100)
SEED = 20260929
DOMAINS = ("MIT", "XJTU", "NASA")
WIDTHS = {"MIT": 51, "XJTU": 46, "NASA": 44}


def charge_rates(metadata):
    """The source policy sometimes omits the trailing C (e.g. 4C(31%)-5)."""
    if metadata["dataset"] == "MIT":
        match = re.fullmatch(r"([0-9.]+)C\([0-9.]+%\)-([0-9.]+)C?(?:-newstructure)?",
                             metadata.get("charge_protocol") or "")
        if match:
            return [float(match[1]), float(match[2])]
    return metadata.get("charge_rate_steps_c") or [
        metadata.get("charge_rate_c"), metadata.get("charge_rate_c")]


def type_key(metadata):
    dataset = metadata["dataset"]
    if dataset == "MIT":
        rates = charge_rates(metadata)
        if len(rates) != 2 or not np.isfinite(rates).all():
            raise ValueError("MIT requires two known charging rates")
        direction = "constant" if rates[0] == rates[1] else (
            "decreasing" if rates[0] > rates[1] else "increasing")
        return f'MIT/{metadata["group"]}/{direction}'
    if dataset == "XJTU":
        return f'XJTU/{metadata["group"]}'
    if dataset == "NASA":
        return "NASA/2Ah_CC_laboratory"
    raise ValueError("unsupported dataset")


def metadata_vector(metadata, types):
    key = type_key(metadata)
    if key not in types:
        raise ValueError(f"unseen battery type: {key}")
    rates = charge_rates(metadata)
    policy = metadata.get("charge_protocol") or ""
    match = re.search(r"\(([0-9.]+)%\)", policy)
    physical = [rates[0], rates[-1], float(match[1]) / 100 if match else None,
        metadata.get("diagnostic_discharge_rate_c"), metadata.get("discharge_cutoff_v")]
    return np.r_[[float(key == item) for item in types],
                 [np.nan if item is None else float(item) for item in physical]]


def prefix_vector(cycle, raw, metadata, origin, types):
    """Only signal observations <= origin participate; no capacity or EOL input."""
    cycle, raw = np.asarray(cycle), np.asarray(raw, dtype=float)
    if cycle.ndim != 1 or raw.shape != (len(cycle), WIDTHS[metadata["dataset"]]):
        raise ValueError("signal feature schema mismatch")
    if not np.isfinite(cycle).all() or np.any(np.diff(cycle) <= 0):
        raise ValueError("cycle numbers must increase and be finite")
    # Cut before examining signal values: later observations cannot affect inference.
    take = cycle <= origin
    c, x = cycle[take], raw[take]
    minimum = 6 if metadata["dataset"] == "XJTU" else 3
    if len(c) < minimum:
        raise ValueError("insufficient signal history at requested origin")
    if np.isinf(x).any() or not np.isfinite(x).any(axis=1).all():
        raise ValueError("invalid signal history")
    ref = 5 if metadata["dataset"] == "XJTU" else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        initial = np.nanmedian(x[:ref], axis=0)
        last = np.nanmedian(x[-3:], axis=0)
        delta = last - initial
        xx, tt = x[-min(10, len(x)):], c[-min(10, len(x)):].astype(float)
        valid = np.isfinite(xx)
        n = valid.sum(axis=0)
        t_mean = (valid * tt[:, None]).sum(axis=0) / np.maximum(n, 1)
        x_mean = np.nansum(xx, axis=0) / np.maximum(n, 1)
        tc = tt[:, None] - t_mean
        numerator = np.nansum(tc * (xx - x_mean), axis=0)
        denominator = (tc * tc * valid).sum(axis=0)
        slope = np.divide(numerator, denominator, out=np.full(x.shape[1], np.nan),
                          where=(n >= 2) & (denominator > 0)) * 100
    return np.r_[last, initial, delta, slope, origin / 100, c[-1] / 100,
                 len(c) / 100, (c[-1] - c[0]) / 100,
                 metadata_vector(metadata, types)].astype(np.float32)


def query_matrix(prefix, origin, future_cycles):
    future = np.asarray(future_cycles, dtype=float)
    if future.ndim != 1 or not len(future) or not np.isfinite(future).all() or np.any(future <= origin):
        raise ValueError("requested cycles must be finite and later than origin")
    h = future - origin
    return np.column_stack((np.tile(prefix, (len(h), 1)), future / 100,
                            h / 100, np.log1p(h))).astype(np.float32)


def fit_curve_model(cells, ids, leaf, seed=SEED):
    """Labels belong only to explicitly listed training cells."""
    types = sorted({type_key(cells[key]["metadata"]) for key in ids})
    x_parts, y_parts, w_parts, templates = [], [], [], []
    for key in ids:
        cell = cells[key]
        label_cycle, soh = cell["label_cycle"], cell["soh"]
        finite = np.isfinite(soh) & (soh > 0)
        templates.append({"cell_id": key, "type": type_key(cell["metadata"]),
                          "cycle": label_cycle[finite], "soh": soh[finite]})
        parts = []
        for origin in ORIGINS:
            try:
                prefix = prefix_vector(cell["cycle"], cell["raw"], cell["metadata"], origin, types)
            except ValueError as error:
                if "insufficient" in str(error):
                    continue
                raise
            future = np.flatnonzero(finite & (label_cycle > origin))
            if not len(future):
                continue
            # Equal support per cell/origin, not proportional to cycle-life length.
            if len(future) > 96:
                future = future[np.unique(np.linspace(0, len(future) - 1, 96).round().astype(int))]
            parts.append((query_matrix(prefix, origin, label_cycle[future]), soh[future]))
        for x, y in parts:
            x_parts.append(x)
            y_parts.append(y)
            w_parts.append(np.full(len(y), 1 / (len(parts) * len(y))))
    if not x_parts:
        raise ValueError("no training future curves")
    x, y, weight = np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(w_parts)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=leaf,
        max_features=.8, random_state=seed, n_jobs=4)
    model.fit(transformed, y, sample_weight=weight)
    return {"schema": 1, "model": model, "imputer": imputer, "types": types,
            "templates": templates, "train_ids": list(ids), "leaf": leaf,
            "seed": seed, "training_pairs": len(y), "origins": ORIGINS}


def predict_curve(bundle, cycle, raw, metadata, origin, future_cycles, method="tree"):
    prefix = prefix_vector(cycle, raw, metadata, origin, bundle["types"])
    x = query_matrix(prefix, origin, future_cycles)
    key = type_key(metadata)
    examples = [r for r in bundle["templates"] if r["type"] == key and len(r["cycle"])]
    if not examples:
        raise ValueError("no same-type training curves")
    if method == "tree":
        prediction = bundle["model"].predict(bundle["imputer"].transform(x))
    elif method == "type_mean":
        prediction = np.mean([np.interp(future_cycles, r["cycle"], r["soh"]) for r in examples], axis=0)
    else:
        raise ValueError("unknown forecast method")
    support = np.sum([np.asarray(future_cycles) <= r["cycle"][-1] for r in examples], axis=0)
    return np.asarray(prediction), np.asarray(support)
