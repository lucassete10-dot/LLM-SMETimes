"""Within-MIT-Batch-2 exact-80% RUL development baselines.

All exact-80% events in the audited MIT cohort are in Batch 2, so this
experiment cannot establish cross-batch RUL generalization. Features are
signal-only and causal; the event cycle is used as a training target and to
locate retrospective evaluation origins, never as an inference feature.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from battery_smetimes.data.split import cellwise_split
from battery_v2_features import load_cell, load_manifest


AUDIT = Path("battery_models_v2/mit_exact80_endpoint_audit.json")
OUT = Path("battery_models_v3/mit_exact80_rul_stage_a")
FRACTIONS = (.2, .4, .6, .8)
SEED = 2026


def checkpoint(cell, event_cycle, fraction):
    """Last labeled, non-flagged diagnostic by the retrospective life fraction."""
    cutoff = fraction * event_cycle
    eligible = np.flatnonzero((cell.cycle <= cutoff) & cell.soh_available &
                              ~cell.capacity_review_flag)
    if not len(eligible):
        return None
    return int(eligible[-1])


def feature_matrix(cell):
    """Age plus observed V/I/T/Q-prefix features; no capacity or SOH label."""
    return np.column_stack((cell.cycle.astype(np.float32),
                            cell.matrix("signal_history"))).astype(np.float32)


def training_rows(cells, events, ids):
    x, y, weights = [], [], []
    for key in ids:
        cell = cells[key]
        event = events[key]
        eligible = np.flatnonzero((cell.cycle < event) & cell.soh_available &
                                  ~cell.capacity_review_flag)
        indices = eligible[::10]
        if not len(indices):
            raise ValueError(f"no pre-EOL training observations: {key}")
        x.append(feature_matrix(cell)[indices])
        y.append(np.full(len(indices), event, dtype=np.float32))
        weights.append(np.full(len(indices), 1 / (len(ids) * len(indices))))
    return np.concatenate(x), np.concatenate(y), np.concatenate(weights)


def checkpoint_rows(cells, events, ids):
    rows = []
    for key in ids:
        cell = cells[key]
        event = events[key]
        matrix = feature_matrix(cell)
        for fraction in FRACTIONS:
            position = checkpoint(cell, event, fraction)
            if position is None:
                continue
            origin = int(cell.cycle[position])
            if not 0 < origin < event:
                raise ValueError(f"invalid pre-EOL origin: {key}")
            rows.append({"cell_id": key, "fraction": fraction,
                         "origin_cycle": origin, "actual_eol_cycle": event,
                         "x": matrix[position]})
    return rows


def summarize(rows, predicted_eol):
    if len(rows) != len(predicted_eol):
        raise ValueError("prediction length mismatch")
    checkpoints = {}
    for fraction in FRACTIONS:
        slots = [i for i, row in enumerate(rows) if row["fraction"] == fraction]
        actual_eol = np.asarray([rows[i]["actual_eol_cycle"] for i in slots])
        origin = np.asarray([rows[i]["origin_cycle"] for i in slots])
        estimate_eol = np.asarray(predicted_eol)[slots]
        actual_rul = actual_eol - origin
        estimate_rul = np.maximum(0, estimate_eol - origin)
        rul_ape = 100 * np.abs(estimate_rul - actual_rul) / actual_rul
        checkpoints[str(int(100 * fraction))] = {
            "cells": len(slots),
            "eol_mae_cycles": float(np.mean(np.abs(estimate_eol - actual_eol))),
            "eol_mape_percent": float(np.mean(100 * np.abs(estimate_eol - actual_eol) /
                                               actual_eol)),
            "rul_mae_cycles": float(np.mean(np.abs(estimate_rul - actual_rul))),
            "rul_mape_percent": float(np.mean(rul_ape)),
            "rul_median_ape_percent": float(np.median(rul_ape)),
            "rul_p90_ape_percent": float(np.quantile(rul_ape, .9)),
            "rul_worst_ape_percent": float(np.max(rul_ape)),
            "premature_eol_count": int(np.sum(estimate_eol <= origin)),
        }
    return checkpoints


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    events = {row["cell_id"]: int(row["exact80_event_cycle"])
              for row in audit["by_cell"] if row["exact80_event_cycle"] is not None}
    if len(events) != 43 or any(row["group"] != "batch2" for row in audit["by_cell"]
                                 if row["cell_id"] in events):
        raise ValueError("MIT exact-80% event cohort changed; re-audit first")
    records = [row for row in load_manifest()["cells"] if row["cell_id"] in events]
    if len(records) != len(events) or any(row["dataset"] != "MIT" for row in records):
        raise ValueError("MIT event cell/manifest mismatch")
    split = cellwise_split(records, "MIT", seed=SEED)
    cells = {row["cell_id"]: load_cell(row) for row in records}
    x_train, y_train, weights = training_rows(cells, events, split["train"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x_train)
    train = imputer.transform(x_train)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(train)
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    ridge = Ridge(alpha=100).fit(scaler.transform(train), y_train,
                                 sample_weight=weights * len(train))
    trees = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=SEED, n_jobs=-1)
    trees.fit(train, y_train, sample_weight=weights)
    median_life = float(np.median([events[key] for key in split["train"]]))
    evaluations = {}
    for part in ("validation", "test"):
        rows = checkpoint_rows(cells, events, split[part])
        x = imputer.transform(np.stack([row["x"] for row in rows]))
        predictions = {
            "median_train_eol": np.full(len(rows), median_life),
            "ridge": ridge.predict(scaler.transform(x)),
            "extra_trees": trees.predict(x),
        }
        evaluations[part] = {
            name: summarize(rows, estimate) for name, estimate in predictions.items()}
        print(part, {name: round(float(np.mean([score["rul_mape_percent"]
                                              for score in result.values()])), 2)
                     for name, result in evaluations[part].items()}, flush=True)
    selected = min(evaluations["validation"], key=lambda name: np.mean([
        score["rul_mape_percent"]
        for score in evaluations["validation"][name].values()]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    joblib.dump({"imputer": imputer, "scaler": scaler, "ridge": ridge,
                 "extra_trees": trees, "train_median_eol": median_life,
                 "train_cell_ids": split["train"]}, OUT / "models.joblib")
    report = {
        "status": "development only; exact80 events occur in MIT Batch 2 only",
        "cohort": "43 observed events, 91 other MIT cells right-censored/excluded",
        "endpoint_rule": audit["exact80_rule"],
        "checkpoint_definition": "last labeled non-flagged cycle <= fraction of true EOL; fraction is evaluation only",
        "input": "source cycle plus causal 300-second signal-history features; no capacity, SOH, or event target",
        "split_cell_counts": {part: len(ids) for part, ids in split.items()},
        "training_observations": len(train),
        "selection_metric": "validation mean of four checkpoint RUL MAPE values",
        "selected_on_validation": selected,
        "metrics": evaluations,
    }
    (OUT / "development_results.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    print("selected on validation:", selected, flush=True)


if __name__ == "__main__":
    main()
