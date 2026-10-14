"""Fixed-hyperparameter leave-one-dataset-out current-SOH diagnostic.

This is zero-shot across data sources, not an independent final acceptance
test. It uses only the harmonized, causal physical signal representation and
previously established reference capacity. Whole cells stay in one fold.
"""

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_smetimes.features.universal import causal_shared_features


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "battery_models_v3" / "cross_dataset_shared_soh"
DOMAINS = ("NASA", "CALCE", "MIT", "XJTU")
MAX_TRAIN_ROWS_PER_CELL = 128


def load_cells():
    cell_rows = pq.read_table(PROCESSED / "cells.parquet").to_pylist()
    cells = {(row["dataset_id"], row["cell_id"]): row for row in cell_rows}
    if len(cells) != len(cell_rows):
        raise ValueError("duplicate canonical cell key")
    cycles = pq.ParquetFile(PROCESSED / "cycles.parquet")
    features = pq.ParquetFile(PROCESSED / "features.parquet")
    if cycles.num_row_groups != len(cells) or features.num_row_groups != len(cells):
        raise ValueError("canonical row-group count mismatch")
    result = {}
    for i in range(len(cells)):
        labels = cycles.read_row_group(i, columns=["dataset_id", "cell_id",
            "cycle_index", "soh_capacity", "soh_available"]).to_pydict()
        signal = features.read_row_group(i, columns=["dataset_id", "cell_id",
            "cycle_index", "raw_signal_features"]).to_pydict()
        key = (labels["dataset_id"][0], labels["cell_id"][0])
        if (key not in cells or (signal["dataset_id"][0], signal["cell_id"][0]) != key
                or labels["cycle_index"] != signal["cycle_index"]):
            raise ValueError(f"cycle/feature join mismatch in row group {i}")
        record = cells[key]
        raw = np.asarray(signal["raw_signal_features"], dtype=np.float32)
        x = causal_shared_features(raw, key[0], record["reference_capacity_ah"],
                                   record["reference_measurement_count"])
        mask = np.asarray(labels["soh_available"], dtype=bool)
        y = np.asarray([v if v is not None else np.nan
                        for v in labels["soh_capacity"]], dtype=np.float64)
        if not mask.any() or not np.isfinite(y[mask]).all() or np.any(y[mask] <= 0):
            raise ValueError(f"invalid scored SOH: {key}")
        result[key] = (x, y, mask)
    return result


def subsample_indices(mask, maximum):
    indices = np.flatnonzero(mask)
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum).astype(int)]


def training_arrays(cells, held_out):
    train_keys = [key for key in sorted(cells) if key[0] != held_out]
    domain_counts = Counter(key[0] for key in train_keys)
    x_parts, y_parts, weights = [], [], []
    for key in train_keys:
        x, y, mask = cells[key]
        indices = subsample_indices(mask, MAX_TRAIN_ROWS_PER_CELL)
        x_parts.append(x[indices])
        y_parts.append(y[indices])
        weights.append(np.full(len(indices),
            1 / (len(domain_counts) * domain_counts[key[0]] * len(indices))))
    return (np.concatenate(x_parts), np.concatenate(y_parts),
            np.concatenate(weights), train_keys)


def summarize(per_cell):
    values = np.asarray([row["mape_percent"] for row in per_cell])
    absolute = np.asarray([row["mae_soh_percentage_points"] for row in per_cell])
    return {"cell_macro_mape_percent": float(values.mean()),
            "cell_macro_mae_soh_percentage_points": float(absolute.mean()),
            "cell_p90_mape_percent": float(np.percentile(values, 90)),
            "worst_cell_mape_percent": float(values.max()),
            "cells": len(per_cell),
            "scored_diagnostics": int(sum(row["scored_diagnostics"] for row in per_cell))}


def score_domain(cells, domain, model=None, imputer=None):
    per_cell = []
    for key in sorted(cells):
        if key[0] != domain:
            continue
        x, y, mask = cells[key]
        target = y[mask]
        prediction = (np.ones(len(target)) if model is None else
                      model.predict(imputer.transform(x[mask])))
        error = np.abs(prediction - target)
        per_cell.append({"cell_id": key[1], "scored_diagnostics": len(target),
                         "mape_percent": float(np.mean(error / target) * 100),
                         "mae_soh_percentage_points": float(error.mean() * 100),
                         "signed_bias_soh_percentage_points": float(np.mean(prediction - target) * 100)})
    return {"summary": summarize(per_cell), "per_cell": per_cell}


def main():
    cells = load_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"status": "development leave-one-dataset-out diagnostic; not final acceptance",
              "target": "same-cycle capacity SOH, not future trajectory or RUL",
              "input": "aligned discharge-prefix V/I/T physical features and causal calibration deltas",
              "calibration": "per-cell reference established before scored diagnostics",
              "train_balance": "equal weight per source domain, equal per cell within domain",
              "max_train_diagnostics_per_cell": MAX_TRAIN_ROWS_PER_CELL,
              "source_folds": {}}
    for held_out in DOMAINS:
        x, y, weights, train_keys = training_arrays(cells, held_out)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
        model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                    max_features=.8, random_state=2026, n_jobs=-1)
        model.fit(imputer.transform(x), y, sample_weight=weights)
        trained = score_domain(cells, held_out, model, imputer)
        persistence = score_domain(cells, held_out)
        report["source_folds"][held_out] = {
            "train_domains": sorted(set(key[0] for key in train_keys)),
            "train_cells": len(train_keys), "train_rows": len(x),
            "shared_extra_trees": trained, "calibrated_soh_1_baseline": persistence}
        joblib.dump({"imputer": imputer, "model": model,
                     "train_cells": train_keys, "held_out_dataset": held_out,
                     "feature_contract": "universal physical prefix v1"},
                    OUT / f"heldout_{held_out.lower()}.joblib")
        print(held_out, "tree MAPE", trained["summary"]["cell_macro_mape_percent"],
              "baseline", persistence["summary"]["cell_macro_mape_percent"], flush=True)
    (OUT / "development_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
