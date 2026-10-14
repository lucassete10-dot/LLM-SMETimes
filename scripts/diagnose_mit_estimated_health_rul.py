"""Nested cell-cross-fit SOH -> exact-80 RUL on the fixed MIT event cohort.

The RUL model sees predicted health histories. Every training cell's SOH
history is produced by an inner model that never trained on that cell. The
outer test cells are absent from both stages of training and preprocessing.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from battery_v2_features import load_cell, load_manifest
from scripts.diagnose_mit_health_rul import OUT as DIAGNOSTIC, observed_health_features
from train_mit_exact80_rul_stage_a import (
    AUDIT, OUT as BASE, SEED, checkpoint_rows, feature_matrix, summarize,
)


OUT = Path("battery_models_v3/mit_estimated_health_rul")
MODES = ("signal_plus_estimated_soh_logrul", "estimated_soh_logrul")
SOH_ROWS_PER_CELL = 128


def fit_soh(cells, signal, ids):
    x, y, weight = [], [], []
    for key in ids:
        cell = cells[key]
        eligible = np.flatnonzero(cell.soh_available & ~cell.capacity_review_flag)
        if len(eligible) > SOH_ROWS_PER_CELL:
            eligible = eligible[np.linspace(0, len(eligible) - 1, SOH_ROWS_PER_CELL).astype(int)]
        x.append(signal[key][eligible])
        y.append(cell.soh[eligible])
        weight.append(np.full(len(eligible), 1 / (len(ids) * len(eligible))))
    x, y, weight = np.concatenate(x), np.concatenate(y), np.concatenate(weight)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=SEED, n_jobs=4)
    model.fit(imputer.transform(x), y, sample_weight=weight)
    return {"imputer": imputer, "model": model, "train_cells": list(ids)}


def predict_soh(bundle, signal):
    return bundle["model"].predict(bundle["imputer"].transform(signal))


def predicted_health_features(cell, prediction):
    # At inference, do not use true capacity flags or future-label availability
    # to decide which historical predictions the RUL stage can see.
    proxy = SimpleNamespace(cycle=cell.cycle, soh=np.asarray(prediction),
                            soh_available=np.arange(len(cell.cycle)) >= cell.reference_count,
                            capacity_review_flag=np.zeros(len(cell.cycle), dtype=bool))
    return observed_health_features(proxy)


def main():
    prior = json.loads((BASE / "five_fold_cell_oof.json").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    events = {r["cell_id"]: int(r["exact80_event_cycle"]) for r in audit["by_cell"]
              if r["exact80_event_cycle"] is not None}
    records = [r for r in load_manifest()["cells"] if r["cell_id"] in events]
    cells = {r["cell_id"]: load_cell(r) for r in records}
    if len(cells) != 43:
        raise ValueError("exact80 event cohort changed")
    signal = {key: feature_matrix(cell) for key, cell in cells.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, folds, soh_metrics = {mode: [] for mode in MODES}, [], []
    for fold in prior["folds"]:
        number, testing = fold["number"], fold["test_cells"]
        training = sorted(set(cells) - set(testing))
        folder = OUT / f"fold{number}"
        folder.mkdir(parents=True, exist_ok=True)
        oof, inner_splits = {}, []
        for inner, (fit_indices, held_indices) in enumerate(
                KFold(3, shuffle=True, random_state=SEED).split(training), 1):
            fit_ids, held_ids = ([training[i] for i in indices]
                                for indices in (fit_indices, held_indices))
            if set(fit_ids) & (set(held_ids) | set(testing)):
                raise ValueError("nested SOH cell leakage")
            bundle = fit_soh(cells, signal, fit_ids)
            oof.update({key: predict_soh(bundle, signal[key]) for key in held_ids})
            inner_splits.append({"train": fit_ids, "predict": held_ids})
            joblib.dump(bundle, folder / f"soh_inner{inner}.joblib")
        if set(oof) != set(training):
            raise ValueError("incomplete out-of-fold training health estimates")
        outer = fit_soh(cells, signal, training)
        joblib.dump(outer, folder / "soh_outer.joblib")
        oof.update({key: predict_soh(outer, signal[key]) for key in testing})
        for key in testing:
            cell = cells[key]
            keep = cell.soh_available & ~cell.capacity_review_flag & (cell.cycle < events[key])
            soh_metrics.append({"cell_id": key, "fold": number,
                                "pre_eol_current_soh_mape_percent": float(np.mean(
                                    100 * np.abs(oof[key][keep] - cell.soh[keep]) / cell.soh[keep]))})
        np.savez_compressed(folder / "soh_predictions.npz", **oof)
        matrices = {}
        for key in training + testing:
            health = predicted_health_features(cells[key], oof[key])
            matrices[key] = {"estimated_soh_logrul": health,
                             "signal_plus_estimated_soh_logrul": np.column_stack((signal[key], health[:, 1:]))}
        indices = {key: np.flatnonzero(
            (cells[key].cycle < events[key]) & cells[key].soh_available &
            ~cells[key].capacity_review_flag)[::10] for key in training}
        target = np.concatenate([np.log1p(events[key] - cells[key].cycle[indices[key]])
                                 for key in training])
        weights = np.concatenate([np.full(len(indices[key]), 1 / (len(training) * len(indices[key])))
                                  for key in training])
        rows = checkpoint_rows(cells, events, testing)
        result = {"fold": number, "train_cells": training, "test_cells": testing,
                  "inner_soh_splits": inner_splits, "metrics": {}}
        for mode in MODES:
            x = np.concatenate([matrices[key][mode][indices[key]] for key in training])
            imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
            model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                        max_features=.8, random_state=SEED, n_jobs=4)
            model.fit(imputer.transform(x), target, sample_weight=weights)
            test_x = np.stack([matrices[r["cell_id"]][mode][np.searchsorted(
                cells[r["cell_id"]].cycle, r["origin_cycle"])] for r in rows])
            eol = np.maximum(0, np.expm1(model.predict(imputer.transform(test_x)))) + np.asarray([
                r["origin_cycle"] for r in rows])
            result["metrics"][mode] = summarize(rows, eol)
            predictions[mode].extend({**{key: row[key] for key in (
                "cell_id", "fraction", "origin_cycle", "actual_eol_cycle")},
                "predicted_eol_cycle": float(value)} for row, value in zip(rows, eol))
            joblib.dump({"imputer": imputer, "model": model, "train_cells": training,
                         "mode": mode, "target": "log1p_rul", "soh_model": "soh_outer.joblib"},
                        folder / f"{mode}.joblib")
        folds.append(result)
        print("fold", number, {mode: round(result["metrics"][mode]["80"]["rul_mape_percent"], 2)
                               for mode in MODES}, flush=True)
    if any(len(rows) != 172 or len({(r["cell_id"], r["fraction"]) for r in rows}) != 172
           for rows in predictions.values()):
        raise ValueError("incomplete nested OOF RUL predictions")
    aggregate = {mode: summarize(rows, [r["predicted_eol_cycle"] for r in rows])
                 for mode, rows in predictions.items()}
    report = {"status": "nested whole-cell SOH-to-RUL development experiment; reused 43 MIT Batch-2 events",
              "input": "source cycle, observed 300s signal prefix, causal model-estimated SOH history",
              "outer_folds": 5, "inner_soh_folds": 3, "soh_train_rows_per_cell_max": SOH_ROWS_PER_CELL,
              "seed": SEED, "folds": folds, "aggregate": aggregate,
              "soh_pre_eol_cell_macro_mape_percent": float(np.mean([
                  r["pre_eol_current_soh_mape_percent"] for r in soh_metrics])),
              "soh_metrics_by_cell": soh_metrics,
              "oof_predictions": predictions,
              "prior_input_diagnostic": str(DIAGNOSTIC / "diagnostic.json"),
              "limits": "event-only single batch; no calibrated uncertainty or full trajectory-consistency claim; 91 censored cells not scored"}
    (OUT / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("RUL MAPE", {mode: {f: round(s["rul_mape_percent"], 2) for f, s in values.items()}
                       for mode, values in aggregate.items()}, flush=True)
    print("SOH MAPE", report["soh_pre_eol_cell_macro_mape_percent"], flush=True)


if __name__ == "__main__":
    main()
