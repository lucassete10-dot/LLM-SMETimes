"""Does observed SOH history resolve the exact-80 RUL information bottleneck?

This controlled diagnostic uses full-discharge SOH through the observation
origin. It is not an online signal-only claim or a mathematical upper bound.
The same five held-out-cell folds, origins and event cohort are reused.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_cell, load_manifest
from train_mit_exact80_rul_stage_a import (
    AUDIT, OUT as BASE, SEED, checkpoint_rows, feature_matrix, summarize,
)


OUT = Path("battery_models_v3/mit_health_information_diagnostic")
MODES = ("signal_eol_reproduction", "signal_logrul",
         "signal_plus_observed_soh_logrul", "observed_soh_logrul")
HEALTH_NAMES = ("source_cycle", "observed_current_soh", "median_last5_soh",
                "soh_slope_per_cycle_20", "soh_slope_per_cycle_50",
                "soh_slope_per_cycle_100", "soh_std_last20_cycles")


def observed_health_features(cell):
    cycle = np.asarray(cell.cycle, dtype=float)
    soh = np.asarray(cell.soh, dtype=float)
    valid = np.asarray(cell.soh_available) & ~np.asarray(cell.capacity_review_flag)
    features = np.full((len(cycle), len(HEALTH_NAMES)), np.nan)
    features[:, 0] = cycle
    available = []
    for index, origin in enumerate(cycle):
        if valid[index] and np.isfinite(soh[index]):
            available.append(index)
        if not available:
            continue
        prior = np.asarray(available, dtype=int)
        features[index, 1] = soh[prior[-1]]
        features[index, 2] = np.median(soh[prior[-5:]])
        for column, window in enumerate((20, 50, 100), 3):
            keep = prior[cycle[prior] >= origin - window]
            if len(keep) >= 3:
                x = cycle[keep] - np.mean(cycle[keep])
                features[index, column] = x @ soh[keep] / (x @ x)
        keep = prior[cycle[prior] >= origin - 20]
        if len(keep):
            features[index, 6] = np.std(soh[keep])
    return features.astype(np.float32)


def feature_variants(cell):
    signal = feature_matrix(cell)
    health = observed_health_features(cell)
    return {"signal_eol_reproduction": signal, "signal_logrul": signal,
            "signal_plus_observed_soh_logrul": np.column_stack((signal, health[:, 1:])),
            "observed_soh_logrul": health}


def main():
    prior = json.loads((BASE / "five_fold_cell_oof.json").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    events = {r["cell_id"]: int(r["exact80_event_cycle"]) for r in audit["by_cell"]
              if r["exact80_event_cycle"] is not None}
    records = [r for r in load_manifest()["cells"] if r["cell_id"] in events]
    cells = {r["cell_id"]: load_cell(r) for r in records}
    if len(cells) != 43 or any(c.group != "batch2" for c in cells.values()):
        raise ValueError("audited MIT event cohort changed")
    matrices = {key: feature_variants(cell) for key, cell in cells.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = {mode: [] for mode in MODES}
    folds = []
    for fold in prior["folds"]:
        number, testing = fold["number"], fold["test_cells"]
        training = sorted(set(cells) - set(testing))
        if set(training) & set(testing):
            raise ValueError("cell split overlap")
        rows = checkpoint_rows(cells, events, testing)
        indices = {}
        y_parts, eol_parts, w_parts = [], [], []
        for key in training:
            cell = cells[key]
            keep = np.flatnonzero((cell.cycle < events[key]) & cell.soh_available &
                                  ~cell.capacity_review_flag)[::10]
            indices[key] = keep
            y_parts.append(np.log1p(events[key] - cell.cycle[keep]))
            eol_parts.append(np.full(len(keep), events[key], dtype=np.float32))
            w_parts.append(np.full(len(keep), 1 / (len(training) * len(keep))))
        target, weights = np.concatenate(y_parts), np.concatenate(w_parts)
        eol_target = np.concatenate(eol_parts)
        fold_result = {"fold": number, "train_cells": training, "test_cells": testing,
                       "metrics": {}}
        for mode in MODES:
            x = np.concatenate([matrices[key][mode][indices[key]] for key in training])
            imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
            model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                        max_features=.8, random_state=SEED, n_jobs=4)
            model.fit(imputer.transform(x),
                      eol_target if mode == "signal_eol_reproduction" else target,
                      sample_weight=weights)
            test_x = np.stack([matrices[row["cell_id"]][mode][np.searchsorted(
                cells[row["cell_id"]].cycle, row["origin_cycle"])] for row in rows])
            prediction = model.predict(imputer.transform(test_x))
            if mode == "signal_eol_reproduction":
                eol = prediction
                stored = {(r["cell_id"], r["fraction"]): r["predicted_eol_cycle"]
                          for r in prior["oof_predictions"]["extra_trees"]}
                difference = np.abs(eol - np.asarray([
                    stored[(r["cell_id"], r["fraction"])] for r in rows]))
                if difference.max() > 1e-7:
                    raise ValueError("original signal/EOL baseline failed reproduction")
                fold_result["baseline_reproduction_max_abs_cycles"] = float(difference.max())
            else:
                remaining = np.maximum(0, np.expm1(prediction))
                eol = remaining + np.asarray([row["origin_cycle"] for row in rows])
            fold_result["metrics"][mode] = summarize(rows, eol)
            predictions[mode].extend({**{key: row[key] for key in (
                "cell_id", "fraction", "origin_cycle", "actual_eol_cycle")},
                "predicted_eol_cycle": float(value)} for row, value in zip(rows, eol))
            joblib.dump({"model": model, "imputer": imputer, "mode": mode,
                         "train_cells": training, "health_features": HEALTH_NAMES,
                         "requires_measured_soh": mode in (
                             "signal_plus_observed_soh_logrul", "observed_soh_logrul"),
                         "target": "absolute_eol" if mode == "signal_eol_reproduction" else "log1p_rul"},
                        OUT / f"fold{number}_{mode}.joblib")
        folds.append(fold_result)
        print("fold", number, {mode: round(fold_result["metrics"][mode]["80"]["rul_mape_percent"], 2)
                               for mode in MODES}, flush=True)
    aggregate = {mode: summarize(rows, [r["predicted_eol_cycle"] for r in rows])
                 for mode, rows in predictions.items()}
    expected = {(r["cell_id"], r["origin_cycle"], r["actual_eol_cycle"], r["fraction"])
                for r in prior["oof_predictions"]["extra_trees"]}
    if any(len(rows) != 172 or {
            (r["cell_id"], r["origin_cycle"], r["actual_eol_cycle"], r["fraction"])
            for r in rows} != expected for rows in predictions.values()):
        raise ValueError("predictions do not match all 43 prior cells and four origins")
    report = {
        "status": "measured-history input diagnostic on reused MIT Batch-2 development folds",
        "warning": "measured SOH requires full observed discharge; not online signal-only; not a mathematical upper bound",
        "input_contract": "current and past diagnostics only; true event used only for training target and retrospective scoring origins",
        "selection": "three fixed variants; no test-based hyperparameter selection",
        "target": "absolute EOL for reproduction; log1p(remaining cycles to confirmed 80%-of-own-reference event) for other variants",
        "health_names": HEALTH_NAMES, "right_censored_cells_not_point_scored": 91,
        "seed": SEED, "folds": folds, "aggregate": aggregate,
        "prior_signal_eol_and_median_baselines": prior["aggregate"],
        "oof_predictions": predictions,
    }
    (OUT / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("RUL MAPE", {mode: {f: round(s["rul_mape_percent"], 2) for f, s in scores.items()}
                       for mode, scores in aggregate.items()}, flush=True)


if __name__ == "__main__":
    main()
