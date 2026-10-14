"""Held-out 80%-SOH RUL ablation using the causal time-gated signal view.

This does not use measured SOH, capacity, or the eventual endpoint as input.
Censored cells have no regression label and are excluded from scored targets.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from paper_inspired_battery_v2 import time_gated_matrix
from train_rul_v2 import eligible
from train_soh_v2 import outer_splits


OUT = Path("battery_models_v2/paper_inspired_soh_challenger/rul_development_cv.json")
KINDS = ("gated_log", "gated_raw", "fixed_half")


def features(cell):
    age = cell.cycle.astype(np.float32)
    return np.column_stack((time_gated_matrix(cell), age, np.log1p(age)))


def fit(cells, ids, matrices, kind):
    parts = [[], [], []]
    active = [key for key in ids if eligible(cells[key]).any()]
    for key in active:
        cell = cells[key]
        keep = eligible(cell)
        x = matrices[key][keep][::5]
        y = cell.rul_cycles[keep][::5]
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype(np.float64)
        weight = (weight / weight.sum())[::5] / len(active)
        for target, value in zip(parts, (x, y, weight)):
            target.append(value)
    x, y, weights = [np.concatenate(part) for part in parts]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=8,
                                max_features=.8, random_state=2026, n_jobs=-1)
    if kind == "gated_log":
        y = np.log1p(y)
    model.fit(imputer.transform(x), y, sample_weight=weights)
    return imputer, model


def predict(bundle, matrix, kind):
    imputer, model = bundle
    result = model.predict(imputer.transform(matrix))
    if kind == "gated_log":
        result = np.expm1(result)
    return np.maximum(0, result)


def baseline_oof(domain, cells):
    path = Path("battery_models_v2/rul") / domain.lower() / "development_oof_predictions.npz"
    with np.load(path) as data:
        rows = {name: data[name].copy() for name in data.files}
    result = {key: np.full(len(cell.cycle), np.nan, dtype=np.float64)
              for key, cell in cells.items() if cell.dataset == domain and cell.endpoint_cycle is not None}
    for key, cycle, estimate in zip(rows["cell_id"], rows["cycle"], rows["estimate"]):
        key = str(key)
        index = np.searchsorted(cells[key].cycle, cycle)
        if index == len(cells[key].cycle) or cells[key].cycle[index] != cycle:
            raise ValueError(f"stored baseline RUL does not align: {key}, {cycle}")
        result[key][index] = estimate
    return result


def score(cells, ids, predictions, checkpoint=50):
    rows = []
    for key in ids:
        cell = cells[key]
        keep = eligible(cell)
        if not keep.any():
            continue
        estimate, actual = predictions[key][keep], cell.rul_cycles[keep]
        if not np.isfinite(estimate).all():
            raise ValueError(f"non-finite predictions for {key}")
        at = np.flatnonzero(keep & (cell.cycle == checkpoint))
        eol_error = (float(100 * abs(cell.cycle[at[0]] + predictions[key][at[0]] -
                                cell.endpoint_cycle) / cell.endpoint_cycle)
                     if len(at) else None)
        rows.append({"cell_id": key, "mae_cycles": float(np.mean(abs(estimate - actual))),
                     "eol_mape_at_50_percent": eol_error})
    at50 = [row["eol_mape_at_50_percent"] for row in rows
            if row["eol_mape_at_50_percent"] is not None]
    return {"cells": len(rows), "cell_macro_mae_cycles": float(np.mean([
                row["mae_cycles"] for row in rows])),
            "worst_cell_mae_cycles": float(max(row["mae_cycles"] for row in rows)),
            "eol_mape_at_50_percent": float(np.mean(at50)) if at50 else None,
            "cells_at_50": len(at50), "by_cell": rows}


def evaluate(domains):
    manifest, cells = load_all()
    report = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
              "status": "development challenger; selected RUL model unchanged",
              "method": "direct RUL from causal cycle-gap-aware signal, raw and log targets",
              "domains": {}}
    for domain in domains:
        ids = [row["cell_id"] for row in manifest["cells"]
               if row["dataset"] == domain and cells[row["cell_id"]].endpoint_cycle is not None]
        matrices = {key: features(cells[key]) for key in ids}
        baseline = baseline_oof(domain, cells)
        oof = {kind: {} for kind in KINDS}
        folds = []
        for held_out, train_ids, val_ids in outer_splits(manifest["cells"], domain):
            training = [key for key in train_ids if key in matrices]
            validation = [key for key in val_ids if key in matrices]
            if not validation:
                continue
            row = {"held_out": held_out,
                   "baseline": score(cells, validation, baseline)}
            fold_predictions = {}
            for kind in KINDS[:2]:
                bundle = fit(cells, training, matrices, kind)
                pred = {key: predict(bundle, matrices[key], kind) for key in validation}
                fold_predictions[kind] = pred
                oof[kind].update(pred)
                row[kind] = score(cells, validation, pred)
            half = {key: .5 * (fold_predictions["gated_log"][key] +
                                fold_predictions["gated_raw"][key]) for key in validation}
            oof["fixed_half"].update(half)
            row["fixed_half"] = score(cells, validation, half)
            folds.append(row)
            print(domain, held_out, {name: round(row[name]["cell_macro_mae_cycles"], 1)
                                     for name in ("baseline", *KINDS)}, flush=True)
        combined = {name: score(cells, ids, baseline if name == "baseline" else oof[name])
                    for name in ("baseline", *KINDS)}
        # Persist the exact held-out predictions used by the fold report so
        # downstream full-curve audits do not silently substitute baseline RUL.
        row_id = np.concatenate([np.full(int(eligible(cells[key]).sum()), key)
                                 for key in ids])
        row_cycle = np.concatenate([cells[key].cycle[eligible(cells[key])]
                                    for key in ids])
        np.savez_compressed(
            OUT.parent / f'{domain.lower()}_rul_development_oof.npz',
            cell_id=row_id, cycle=row_cycle,
            baseline=np.concatenate([baseline[key][eligible(cells[key])] for key in ids]),
            **{kind: np.concatenate([oof[kind][key][eligible(cells[key])]
                                     for key in ids]) for kind in KINDS})
        report["domains"][domain] = {"summary": combined, "folds": folds}
        print("SUMMARY", domain, {name: round(value["cell_macro_mae_cycles"], 1)
                                   for name, value in combined.items()}, flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", OUT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", choices=("NASA", "CALCE", "MIT", "XJTU"),
                        default=["NASA", "CALCE", "MIT", "XJTU"])
    evaluate(parser.parse_args().domains)
