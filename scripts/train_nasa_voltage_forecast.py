"""One-factor NASA electrical-state challenger on the frozen four-cell folds."""

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from battery_smetimes.forecast import ORIGINS
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data, metrics

SOURCE = Path("battery_models_v4/same_type_forecast/runs/20260929T060454_564594Z")
OUT = Path("battery_models_v4/nasa_voltage_forecast")


def voltage_prefix(cell, origin):
    keep = cell["cycle"] <= origin
    if keep.sum() < 3:
        raise ValueError("insufficient NASA voltage history")
    observed = cell["raw"][keep]
    # NASA 44-column signal contract: 0..10 are discharge voltage at 0..1000 s.
    voltage = float(np.median(observed[-3:, 10]))
    if not np.isfinite(voltage):
        raise ValueError("non-finite NASA discharge voltage at 1000 s")
    return voltage


def input_matrix(voltage, origin, future_cycle):
    future = np.asarray(future_cycle)
    return np.column_stack((np.full(len(future), voltage),
                            np.full(len(future), origin / 100),
                            (future - origin) / 100)).astype(float)


def fit(cells, ids):
    x_parts, y_parts, weights = [], [], []
    for key in ids:
        cell = cells[key]
        parts = []
        for origin in ORIGINS:
            future = np.isfinite(cell["soh"]) & (cell["label_cycle"] > origin)
            index = np.flatnonzero(future)
            if len(index) > 96:
                index = index[np.unique(np.linspace(0, len(index) - 1, 96).round().astype(int))]
            if len(index):
                v = voltage_prefix(cell, origin)
                parts.append((input_matrix(v, origin, cell["label_cycle"][index]), cell["soh"][index]))
        for x, y in parts:
            x_parts.append(x)
            y_parts.append(y)
            weights.append(np.full(len(y), 1 / (len(ids) * len(parts) * len(y))))
    x, y, w = np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(weights)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=.1).fit(scaler.transform(x), y, sample_weight=w * len(y))
    return {"model": model, "scaler": scaler, "train_ids": ids, "origin_grid": ORIGINS,
            "dataset": "NASA", "origins": ORIGINS, "model_schema": "nasa_voltage_ridge_v1",
            "training_support_end_cycles": [int(cells[key]["label_cycle"][np.isfinite(cells[key]["soh"])][-1]) for key in ids],
            "input": "observed 1000 s discharge voltage near origin + source-cycle horizon",
            "target_training_rows": len(y), "weight_per_cell": True}


def predict(bundle, cell, origin, future_cycle):
    voltage = voltage_prefix(cell, origin)
    matrix = input_matrix(voltage, origin, future_cycle)
    return np.clip(bundle["model"].predict(bundle["scaler"].transform(matrix)), 0, 1.2)


def evaluate(bundle, cells, ids, folder):
    rows, arrays = [], {}
    for key in ids:
        cell = cells[key]
        for origin in ORIGINS:
            future = np.isfinite(cell["soh"]) & (cell["label_cycle"] > origin)
            fc, truth = cell["label_cycle"][future], cell["soh"][future]
            if len(fc) < 2:
                continue
            estimate = predict(bundle, cell, origin, fc)
            row = {"cell_id": key, "origin": origin, "future_points": len(fc),
                   **metrics(truth, estimate), "last_observed_cycle": int(fc[-1]),
                   "voltage_t1000_v": voltage_prefix(cell, origin)}
            rows.append(row)
            arrays[f"{key}__{origin}__cycle"] = fc
            arrays[f"{key}__{origin}__truth"] = truth
            arrays[f"{key}__{origin}__prediction"] = estimate
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "metrics.json", {"rows": rows,
        "origin100_cell_macro_mape_percent": float(np.mean([r["mape_percent"] for r in rows if r["origin"] == 100]))})
    np.savez_compressed(folder / "predictions.npz", **arrays)
    return rows


def main():
    config = json.loads((SOURCE / "config.json").read_text())
    _, cells = load_data(Path(config["data_folder"]), config["integrity_file"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / "runs" / stamp
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "config.json", {"source_run": str(SOURCE.resolve()),
        "source_splits_sha256": sha(SOURCE / "splits.json"),
        "changed_hypothesis": "NASA discharge voltage at 1000 s encodes cell-specific SOH level absent from type-mean/tree forecast",
        "candidate": "standardized ridge voltage+horizon, fixed alpha 0.1",
        "selection": "validation full-future MAPE from origin 100; prior baseline remains available",
        "status": "development diagnostic on already-inspected NASA cells; not blind acceptance"})
    try:
        folds = []
        for split in json.loads((SOURCE / "splits.json").read_text()):
            if split["dataset"] != "NASA":
                continue
            folder = run / split["name"]
            folder.mkdir()
            save_json(folder / "split.json", split)
            candidate = fit(cells, split["train"])
            joblib.dump(candidate, folder / "validation_model.joblib", compress=3)
            val = evaluate(candidate, cells, split["validation"], folder / "validation")
            candidate_val = float(np.mean([r["mape_percent"] for r in val if r["origin"] == 100]))
            prior = json.loads((SOURCE / split["name"] / "result.json").read_text())
            prior_val = float(prior["validation_scores"][prior["selected"]])
            route = "voltage_ridge" if candidate_val < prior_val else "prior_selected"
            final = fit(cells, split["train"] + split["validation"])
            joblib.dump(final, folder / "refit_model.joblib", compress=3)
            test = evaluate(final, cells, split["test"], folder / "test")
            with np.load(folder / "test/predictions.npz") as arrays, np.load(SOURCE / split["name"] / "test/predictions.npz") as previous:
                key = split["test"][0]
                fc = arrays[f"{key}__100__cycle"]
                np.testing.assert_array_equal(fc, previous[f"{key}__100__cycle"])
                truth = arrays[f"{key}__100__truth"]
                ridge = arrays[f"{key}__100__prediction"]
                old = previous[f'{key}__100__{prior["selected_method"]}']
                selected = ridge if route == "voltage_ridge" else old
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.plot(fc, truth, "k-", label="Observed SOH", lw=2)
                ax.plot(fc, ridge, color="#1976a3", label="Voltage candidate", lw=1.5)
                ax.plot(fc, old, color="#c28320", ls="--", label="Prior selected", lw=1.5)
                ax.set(xlabel="Source cycle (forecast issued at 100)", ylabel="SOH",
                       title=f'{key} | validation selected {route}')
                ax.legend()
                ax.grid(alpha=.2)
                fig.tight_layout()
                fig.savefig(folder / "comparison.png", dpi=150)
                plt.close(fig)
                result = {"fold": split["name"], "test_cell": key,
                    "candidate_validation_mape_percent": candidate_val,
                    "prior_validation_mape_percent": prior_val, "selected_by_validation": route,
                    "candidate_test_mape_percent": metrics(truth, ridge)["mape_percent"],
                    "prior_test_mape_percent": metrics(truth, old)["mape_percent"],
                    "validation_selected_test_mape_percent": metrics(truth, selected)["mape_percent"],
                    "prediction_cycle_count": len(fc)}
            save_json(folder / "result.json", result)
            folds.append(result)
            save_json(run / "progress.json", {"completed": len(folds), "expected": 4})
            print("SAVED", key, "val selected", route, "test", round(result["validation_selected_test_mape_percent"], 3), flush=True)
        save_json(run / "result.json", {"folds": folds,
            "prior_four_cell_macro_mape_percent": float(np.mean([r["prior_test_mape_percent"] for r in folds])),
            "candidate_four_cell_macro_mape_percent": float(np.mean([r["candidate_test_mape_percent"] for r in folds])),
            "validation_selected_four_cell_macro_mape_percent": float(np.mean([r["validation_selected_test_mape_percent"] for r in folds]))})
        # Exploratory deployment artifact: all four known NASA cells become training
        # cells only after the frozen leave-one-cell-out development comparison.
        full_model = fit(cells, ["B0005", "B0006", "B0007", "B0018"])
        joblib.dump(full_model, run / "all_four_cells_exploratory_model.joblib", compress=3)
        save_json(run / "all_four_cells_exploratory_model.json", {
            "training_cells": full_model["train_ids"],
            "method": "single fixed voltage-ridge challenger",
            "status": "exploratory; no independent fifth NASA cell tested",
            "trained_after_development_evaluation": True})
        save_json(OUT / "latest.json", {"run": str(run.resolve())})
        print("COMPLETE", run, flush=True)
    except BaseException:
        save_json(run / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
