"""Five-fold whole-cell OOF check of MIT Batch-2 exact-80% RUL baselines."""

import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_cell, load_manifest
from train_mit_exact80_rul_stage_a import (
    AUDIT, OUT, SEED, checkpoint_rows, summarize, training_rows,
)


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    events = {row["cell_id"]: int(row["exact80_event_cycle"])
              for row in audit["by_cell"] if row["exact80_event_cycle"] is not None}
    if len(events) != 43:
        raise ValueError("MIT exact80 cohort changed")
    records = [row for row in load_manifest()["cells"] if row["cell_id"] in events]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    ids = sorted(cells)
    oof = defaultdict(list)
    folds = []
    for number, (train_indices, test_indices) in enumerate(
            KFold(n_splits=5, shuffle=True, random_state=SEED).split(ids), 1):
        train_ids = [ids[i] for i in train_indices]
        test_ids = [ids[i] for i in test_indices]
        x_train, y_train, weight = training_rows(cells, events, train_ids)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x_train)
        x = imputer.transform(x_train)
        scaler = RobustScaler(quantile_range=(10, 90)).fit(x)
        scaler.scale_ = np.maximum(scaler.scale_, .02)
        ridge = Ridge(alpha=100).fit(scaler.transform(x), y_train,
                                     sample_weight=weight * len(x))
        tree = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                   max_features=.8, random_state=SEED, n_jobs=-1)
        tree.fit(x, y_train, sample_weight=weight)
        median_life = float(np.median([events[key] for key in train_ids]))
        rows = checkpoint_rows(cells, events, test_ids)
        test_x = imputer.transform(np.stack([row["x"] for row in rows]))
        predictions = {
            "median_train_eol": np.full(len(rows), median_life),
            "ridge": ridge.predict(scaler.transform(test_x)),
            "extra_trees": tree.predict(test_x),
        }
        fold = {"number": number, "train_cells": len(train_ids),
                "test_cells": test_ids, "metrics": {}}
        for name, estimate in predictions.items():
            fold["metrics"][name] = summarize(rows, estimate)
            oof[name].extend({
                "cell_id": row["cell_id"],
                "fraction": row["fraction"],
                "origin_cycle": row["origin_cycle"],
                "actual_eol_cycle": row["actual_eol_cycle"],
                "predicted_eol_cycle": float(prediction),
            } for row, prediction in zip(rows, estimate))
        folds.append(fold)
        print("fold", number, "test", len(test_ids),
              "median80", round(fold["metrics"]["median_train_eol"]["80"]
                                ["rul_mape_percent"], 1), flush=True)
    aggregate = {name: summarize(rows, [row["predicted_eol_cycle"] for row in rows])
                 for name, rows in oof.items()}
    for name, rows in oof.items():
        if len(rows) != 43 * 4 or len({(row["cell_id"], row["fraction"])
                                      for row in rows}) != len(rows):
            raise ValueError(f"incomplete whole-cell OOF: {name}")
    report = {
        "status": "development 5-fold whole-cell OOF; 43 observed Batch-2 events only",
        "seed": SEED, "folds": folds, "aggregate": aggregate,
        "caveat": "91 right-censored MIT cells excluded; no cross-batch exact80 verification",
        "oof_predictions": oof,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "five_fold_cell_oof.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
    print("OOF RUL MAPE", {name: {fraction: round(score["rul_mape_percent"], 1)
                                        for fraction, score in checkpoints.items()}
                             for name, checkpoints in aggregate.items()}, flush=True)


if __name__ == "__main__":
    main()
