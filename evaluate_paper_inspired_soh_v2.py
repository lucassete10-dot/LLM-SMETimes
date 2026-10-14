"""Development CV of a causal time-gated, two-view SOH challenger.

The 50/50 fusion is fixed before inspecting the outer folds. Stored baseline
OOF predictions use exactly the same held-out cells/groups as this experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from evaluate_relative_soh_features_v2 import baseline_oof, fit_view, predict_view
from paper_inspired_battery_v2 import time_gated_matrix
from train_soh_v2 import outer_splits


OUT = Path("battery_models_v2/paper_inspired_soh_challenger")


def fit_gated(cells, ids, matrices):
    parts = [[], [], []]
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        if not keep.any():
            continue
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype(np.float64)
        weight /= weight.sum() * len(ids)
        for target, value in zip(parts, (matrices[key][keep], cell.soh[keep], weight)):
            target.append(value)
    x, y, weights = [np.concatenate(part) for part in parts]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=8,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weights)
    return imputer, model


def summary(cells, ids, predictions):
    rows = []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        target, estimate = cell.soh[keep], predictions[key][keep]
        if not len(target):
            continue
        rows.append({"cell_id": key, "mae": float(np.mean(np.abs(target - estimate))),
                     "mape_percent": float(100 * np.mean(np.abs(target - estimate) / target))})
    return {"cells": len(rows), "cell_macro_mae": float(np.mean([r["mae"] for r in rows])),
            "cell_macro_mape_percent": float(np.mean([r["mape_percent"] for r in rows])),
            "worst_cell_mae": float(max(r["mae"] for r in rows)), "by_cell": rows}


def evaluate(domains):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cells = load_all()
    path = OUT / "development_cv.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
              "status": "development challenger; frozen production models unchanged",
              "method": "raw-signal expert + causal cycle-gap-gated multiscale expert; fixed 50/50 fusion",
              "validation": "whole-cell NASA/CALCE; whole-batch MIT/XJTU",
              "domains": {}}
    for domain in domains:
        ids = [row["cell_id"] for row in manifest["cells"] if row["dataset"] == domain]
        matrices = {key: time_gated_matrix(cells[key]) for key in ids}
        baseline = baseline_oof(cells, domain)
        folds = []
        oof = {name: {} for name in ("raw", "gated", "fusion")}
        for held_out, training, validation in outer_splits(manifest["cells"], domain):
            raw_bundle = fit_view(cells, training, "raw_only")
            gate_bundle = fit_gated(cells, training, matrices)
            pred = {"baseline": {key: baseline[key] for key in validation},
                    "raw": {key: predict_view(raw_bundle, cells[key], "raw_only")
                            for key in validation},
                    "gated": {key: gate_bundle[1].predict(gate_bundle[0].transform(matrices[key]))
                              for key in validation}}
            pred["fusion"] = {key: .5 * (pred["raw"][key] + pred["gated"][key])
                              for key in validation}
            for name in oof:
                oof[name].update(pred[name])
            folds.append({"held_out": held_out, "train_cells": len(training),
                          "validation_cells": len(validation),
                          **{name: summary(cells, validation, estimate)
                             for name, estimate in pred.items()}})
            print(domain, held_out, {name: round(folds[-1][name]["cell_macro_mae"], 4)
                                     for name in pred}, flush=True)
        combined = {name: summary(cells, ids, baseline if name == "baseline" else oof[name])
                    for name in ("baseline", *oof)}
        split_by_cell = {key: fold['held_out'] for fold in folds
                         for key in [row['cell_id'] for row in manifest['cells']
                                     if row['dataset'] == domain and
                                     row['group'] == fold['held_out']]}
        if domain in ('NASA', 'CALCE'):
            split_by_cell = {key: key for key in ids}
        if set(split_by_cell) != set(ids):
            raise ValueError(f'{domain}: incomplete SOH fold assignment')
        np.savez_compressed(
            OUT / f'{domain.lower()}_development_oof_predictions.npz',
            cell_id=np.concatenate([np.full(len(cells[key].cycle), key) for key in ids]),
            cycle=np.concatenate([cells[key].cycle for key in ids]),
            held_out=np.concatenate([np.full(len(cells[key].cycle), split_by_cell[key])
                                     for key in ids]),
            target=np.concatenate([cells[key].soh for key in ids]),
            soh_available=np.concatenate([cells[key].soh_available for key in ids]),
            raw=np.concatenate([oof['raw'][key] for key in ids]),
            gated=np.concatenate([oof['gated'][key] for key in ids]),
            fusion=np.concatenate([oof['fusion'][key] for key in ids]),
        )
        report["domains"][domain] = {"summary": combined, "folds": folds}
        print("SUMMARY", domain,
              {name: round(result["cell_macro_mape_percent"], 2)
               for name, result in combined.items()}, flush=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", choices=("NASA", "CALCE", "MIT", "XJTU"),
                        default=["NASA", "CALCE", "MIT", "XJTU"])
    evaluate(parser.parse_args().domains)
