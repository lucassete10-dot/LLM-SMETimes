"""Whole-cell MIT exact-80% AFT ablation with and without censored cells.

This is a development experiment on a single event-bearing batch. Censored
cells from other MIT batches contribute lower-bound survival likelihood only;
they are never assigned fabricated event cycles or scored as observed events.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from battery_smetimes.models.survival import (
    LogNormalLifeMLP, conditional_lognormal_nll,
    conditional_lognormal_quantile,
)
from battery_v2_features import load_cell, load_manifest
from train_mit_exact80_rul_stage_a import (
    AUDIT, OUT, SEED, checkpoint_rows, feature_matrix, summarize,
)


SNAPSHOT_AGES = (20, 50, 100, 150, 200, 300, 400, 500, 700)
EPOCHS = 100


def snapshot_rows(cells, audit_rows, selected_ids):
    rows = []
    for key in selected_ids:
        cell = cells[key]
        audit = audit_rows[key]
        event = audit["exact80_event_cycle"] is not None
        observed = int(audit["exact80_event_cycle"] if event else
                       audit["observed_through_cycle"])
        matrix = feature_matrix(cell)
        positions = []
        for age in SNAPSHOT_AGES:
            eligible = np.flatnonzero((cell.cycle <= age) & (cell.cycle < observed) &
                                      cell.soh_available & ~cell.capacity_review_flag)
            if len(eligible):
                positions.append(int(eligible[-1]))
        positions = sorted(set(positions))
        if not positions:
            raise ValueError(f"no survival snapshots: {key}")
        for position in positions:
            rows.append({"cell_id": key, "origin_cycle": int(cell.cycle[position]),
                         "observed_time": observed, "event": event,
                         "x": matrix[position], "cell_row_count": len(positions)})
    return rows


def fit_survival(rows, *, events_only, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    selected = [row for row in rows if row["event"] or not events_only]
    if not selected or not any(row["event"] for row in selected):
        raise ValueError("no observed events in training population")
    x = np.stack([row["x"] for row in selected])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
    x = imputer.transform(x)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(x)
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    tensor = torch.from_numpy(np.clip(scaler.transform(x), -8, 8).astype(np.float32))
    origin = torch.tensor([row["origin_cycle"] for row in selected], dtype=torch.float32)
    observed = torch.tensor([row["observed_time"] for row in selected], dtype=torch.float32)
    event = torch.tensor([row["event"] for row in selected], dtype=torch.bool)
    event_cells = len({row["cell_id"] for row in selected if row["event"]})
    censor_cells = len({row["cell_id"] for row in selected if not row["event"]})
    weights = torch.tensor([
        (.5 if censor_cells else 1.0) /
        (event_cells * row["cell_row_count"]) if row["event"] else
        .5 / (censor_cells * row["cell_row_count"])
        for row in selected], dtype=torch.float32)
    model = LogNormalLifeMLP(tensor.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=.02)
    for _ in range(EPOCHS):
        model.train()
        mean, sigma = model(tensor)
        loss = (conditional_lognormal_nll(mean, sigma, origin, observed, event) *
                weights).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, imputer, scaler, {
        "train_event_cells": event_cells, "train_censored_cells": censor_cells,
        "train_snapshots": len(selected), "last_train_nll": float(loss.detach()),
        "learned_loglife_sigma": float(torch.nn.functional.softplus(model.raw_sigma).detach()),
    }


def predict_survival(model, imputer, scaler, rows):
    x = imputer.transform(np.stack([row["x"] for row in rows]))
    tensor = torch.from_numpy(np.clip(scaler.transform(x), -8, 8).astype(np.float32))
    origin = torch.tensor([row["origin_cycle"] for row in rows], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        mean, sigma = model(tensor)
        predictions = {
            str(q): conditional_lognormal_quantile(mean, sigma, origin, q).numpy()
            for q in (.05, .5, .95)}
    if not np.all(np.isfinite(predictions["0.5"])):
        raise ValueError("nonfinite survival forecast")
    return predictions


def interval_coverage(rows, predictions):
    result = {}
    for fraction in (.2, .4, .6, .8):
        slots = [i for i, row in enumerate(rows) if row["fraction"] == fraction]
        truth = np.asarray([rows[i]["actual_eol_cycle"] for i in slots])
        low, high = predictions["0.05"][slots], predictions["0.95"][slots]
        result[str(int(100 * fraction))] = {
            "cells": len(slots),
            "nominal_90pct_eol_interval_coverage": float(np.mean((low <= truth) &
                                                                  (truth <= high))),
            "median_eol_interval_width_cycles": float(np.median(high - low)),
        }
    return result


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_rows = {row["cell_id"]: row for row in audit["by_cell"]}
    events = {key: int(row["exact80_event_cycle"])
              for key, row in audit_rows.items() if row["exact80_event_cycle"] is not None}
    censored = sorted(key for key, row in audit_rows.items()
                      if row["exact80_event_cycle"] is None)
    if len(events) != 43 or len(censored) != 91:
        raise ValueError("MIT exact80 event/censor count changed")
    records = [row for row in load_manifest()["cells"]
               if row["dataset"] == "MIT" and row["cell_id"] in audit_rows]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    if len(cells) != 134:
        raise ValueError("MIT manifest/audit cell mismatch")
    ids = sorted(events)
    folds = []
    oof = defaultdict(list)
    for number, (train_indices, test_indices) in enumerate(
            KFold(n_splits=5, shuffle=True, random_state=SEED).split(ids), 1):
        train_event_ids = [ids[index] for index in train_indices]
        heldout_ids = [ids[index] for index in test_indices]
        train = snapshot_rows(cells, audit_rows, train_event_ids + censored)
        test = checkpoint_rows(cells, events, heldout_ids)
        fold = {"number": number, "heldout_cells": heldout_ids, "models": {}}
        for name, events_only in (("aft_events_only", True),
                                  ("aft_with_censored", False)):
            model, imputer, scaler, metadata = fit_survival(
                train, events_only=events_only, seed=SEED + number)
            prediction = predict_survival(model, imputer, scaler, test)
            fold["models"][name] = {
                "training": metadata,
                "point_metrics": summarize(test, prediction["0.5"]),
                "raw_parametric_intervals": interval_coverage(test, prediction),
            }
            oof[name].extend({
                "cell_id": row["cell_id"], "fraction": row["fraction"],
                "origin_cycle": row["origin_cycle"],
                "actual_eol_cycle": row["actual_eol_cycle"],
                "predicted_eol_cycle": float(prediction["0.5"][slot]),
                "lower_5pct_eol_cycle": float(prediction["0.05"][slot]),
                "upper_95pct_eol_cycle": float(prediction["0.95"][slot]),
            } for slot, row in enumerate(test))
        folds.append(fold)
        print("fold", number, {name: round(row["point_metrics"]["80"]
                                              ["rul_mape_percent"], 1)
                                  for name, row in fold["models"].items()}, flush=True)
    aggregate = {}
    for name, rows in oof.items():
        if len(rows) != 43 * 4 or len({(row["cell_id"], row["fraction"])
                                      for row in rows}) != len(rows):
            raise ValueError(f"incomplete OOF predictions: {name}")
        predictions = {quantile: np.asarray([row[field] for row in rows])
                       for quantile, field in (("0.05", "lower_5pct_eol_cycle"),
                                               ("0.5", "predicted_eol_cycle"),
                                               ("0.95", "upper_95pct_eol_cycle"))}
        aggregate[name] = {
            "point_metrics": summarize(rows, predictions["0.5"]),
            "raw_parametric_intervals": interval_coverage(rows, predictions),
        }
    report = {
        "status": "development 5-fold event-cell OOF; 91 censored cells train-only",
        "seed": SEED, "fixed_snapshot_ages_cycles": SNAPSHOT_AGES,
        "epochs": EPOCHS,
        "censoring_note": "right-censored cells contribute T_EO80 > last observed cycle, not an event label",
        "selection_note": "hyperparameters fixed for ablation; results inspected, not final blind test",
        "interval_note": "raw model 90% intervals, not externally calibrated",
        "folds": folds, "aggregate": aggregate, "oof_predictions": oof,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "censored_survival_five_fold_cv.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RUL MAPE", {name: {f: round(row["rul_mape_percent"], 1)
                               for f, row in result["point_metrics"].items()}
                       for name, result in aggregate.items()}, flush=True)
    print(target, flush=True)


if __name__ == "__main__":
    main()
