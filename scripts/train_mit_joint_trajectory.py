"""Couple saved SOH/RUL estimates to full curves and test honest cell intervals.

Every invocation creates a new timestamped run. Models and fold results are
saved immediately, and an interrupted run retains all completed artifacts.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from battery_smetimes.models.trajectory import (
    anchored_soh, cellwise_log_rul_radius, log_rul_interval,
)
from battery_v2_features import load_cell, load_manifest
from scripts.diagnose_mit_estimated_health_rul import (
    OUT as HEALTH, fit_soh, predict_soh, predicted_health_features,
)
from train_mit_exact80_rul_stage_a import AUDIT, SEED, checkpoint_rows, feature_matrix, summarize


OUT = Path("battery_models_v3/mit_joint_trajectory")
SHAPES = (.5, .75, 1., 1.5, 2., 3.)
SHAPE_ORIGINS = (20, 50, 100, 150, 200, 300, 400)
CALIBRATION_CELLS = 10


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape_fit(cells, events, ids, estimates):
    """Fit one shared shape using only training-cell future capacity labels."""
    scores = {}
    for shape in SHAPES:
        per_cell = []
        for key in ids:
            cell = cells[key]
            valid = cell.soh_available & ~cell.capacity_review_flag
            origins = []
            for cutoff in SHAPE_ORIGINS:
                eligible = np.flatnonzero(valid & (cell.cycle <= cutoff) & (cell.cycle < events[key]))
                if not len(eligible) or cutoff >= events[key]:
                    continue
                index = eligible[-1]
                origin = int(cell.cycle[index])
                future = np.flatnonzero(valid & (cell.cycle > origin) & (cell.cycle <= events[key]))
                if not len(future):
                    continue
                estimate = anchored_soh(estimates[key][index], events[key] - origin,
                                        cell.cycle[future] - origin, shape)
                origins.append(float(np.mean(100 * np.abs(estimate - cell.soh[future]) / cell.soh[future])))
            if origins:
                per_cell.append(float(np.mean(origins)))
        scores[str(shape)] = float(np.mean(per_cell))
    selected = min(SHAPES, key=lambda value: scores[str(value)])
    return selected, {"selected_shape": selected, "training_cells": list(ids),
                      "training_curve_mape_by_shape": scores,
                      "training_origin_cycles": list(SHAPE_ORIGINS),
                      "rule": "source-cell labels supervise shape; true source EOL is a training target, never a query input"}


def nested_fit(cells, signals, events, training, calibration, testing, folder):
    """Fit on proper training cells; calibration/test cells train no component."""
    if set(training) & (set(calibration) | set(testing)) or set(calibration) & set(testing):
        raise ValueError("proper training, calibration and test must be disjoint")
    estimates, splits = {}, []
    for number, (a, b) in enumerate(KFold(3, shuffle=True, random_state=SEED).split(training), 1):
        fit_ids, pred_ids = ([training[i] for i in group] for group in (a, b))
        bundle = fit_soh(cells, signals, fit_ids)
        joblib.dump(bundle, folder / f"soh_inner{number}.joblib")
        estimates.update({key: predict_soh(bundle, signals[key]) for key in pred_ids})
        scores = []
        for key in pred_ids:
            keep = cells[key].soh_available & ~cells[key].capacity_review_flag
            scores.append(float(np.mean(100 * np.abs(estimates[key][keep] - cells[key].soh[keep]) / cells[key].soh[keep])))
        splits.append({"train": fit_ids, "predict": pred_ids})
        np.savez_compressed(folder / f"soh_inner{number}_predictions.npz",
                            **{key: estimates[key] for key in pred_ids})
        save_json(folder / f"soh_inner{number}_result.json", {
            "train_cells": fit_ids, "heldout_cells": pred_ids, "seed": SEED,
            "heldout_soh_cell_macro_mape": float(np.mean(scores)), "per_cell_mape": dict(zip(pred_ids, scores))})
    outer = fit_soh(cells, signals, training)
    joblib.dump(outer, folder / "soh_predictor.joblib")
    estimates.update({key: predict_soh(outer, signals[key]) for key in calibration + testing})
    np.savez_compressed(folder / "soh_predictions.npz", **estimates)
    matrices = {key: np.column_stack((signals[key], predicted_health_features(cells[key], estimates[key])[:, 1:]))
                for key in training + calibration + testing}
    x, y, w = [], [], []
    for key in training:
        cell = cells[key]
        eligible = np.flatnonzero(cell.soh_available & ~cell.capacity_review_flag & (cell.cycle < events[key]))[::10]
        x.append(matrices[key][eligible])
        y.append(np.log1p(events[key] - cell.cycle[eligible]))
        w.append(np.full(len(eligible), 1 / (len(training) * len(eligible))))
    x, y, w = np.concatenate(x), np.concatenate(y), np.concatenate(w)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=SEED, n_jobs=4)
    model.fit(imputer.transform(x), y, sample_weight=w)
    bundle = {"imputer": imputer, "model": model, "train_cells": training,
              "calibration_cells": calibration, "test_cells": testing,
              "inner_soh_splits": splits, "seed": SEED, "target": "log1p_rul"}
    joblib.dump(bundle, folder / "rul_predictor.joblib")
    save_json(folder / "fit_complete.json", {"train_cells": training, "calibration_cells": calibration,
              "test_cells": testing, "inner_soh_splits": splits, "seed": SEED,
              "training_rows": len(y), "model": "rul_predictor.joblib"})
    return estimates, matrices, bundle


def predict_rul(rows, cells, matrices, bundle):
    x = np.stack([matrices[r["cell_id"]][np.searchsorted(cells[r["cell_id"]].cycle, r["origin_cycle"])] for r in rows])
    return np.maximum(0, np.expm1(bundle["model"].predict(bundle["imputer"].transform(x))))


def score_curves(rows, rul_predictions, cells, estimates, shape, radius=None):
    result = []
    for row, remaining in zip(rows, rul_predictions):
        key, origin, event = row["cell_id"], row["origin_cycle"], row["actual_eol_cycle"]
        cell = cells[key]
        position = int(np.searchsorted(cell.cycle, origin))
        current = float(estimates[key][position])
        valid = cell.soh_available & ~cell.capacity_review_flag
        future = np.flatnonzero(valid & (cell.cycle > origin) & (cell.cycle <= event))
        h, truth = cell.cycle[future] - origin, cell.soh[future].astype(float)
        forecast = anchored_soh(current, remaining, h, shape)
        linear = anchored_soh(current, remaining, h, 1.)
        scores = {}
        for name, prediction in (("learned_shape", forecast), ("linear_shape", linear),
                                  ("persistence", np.full(len(h), current))):
            scores[name] = float(np.mean(100 * np.abs(prediction - truth) / truth)) if len(h) else None
        fixed = {}
        for name, horizon in (("10", 10), ("50", 50), ("100", 100),
                              ("20pct_remaining", max(1, round(.2 * (event - origin))))):
            slots = np.flatnonzero(h == horizon)
            fixed[name] = float(100 * abs(forecast[slots[0]] - truth[slots[0]]) / truth[slots[0]]) if len(slots) else None
        result.append({**{key: row[key] for key in ("cell_id", "fraction", "origin_cycle", "actual_eol_cycle")},
                       "predicted_rul": float(remaining), "predicted_eol_cycle": float(origin + remaining),
                       "estimated_current_soh": current, "true_current_soh": float(cell.soh[position]),
                       "anchor_projected_above_80": current <= .8001, "shape": shape,
                       "full_remaining_curve_mape": scores, "fixed_horizon_ape_percent": fixed,
                       "future_cycle": cell.cycle[future].tolist(), "future_true_soh": truth.tolist(),
                       "future_predicted_soh": forecast.tolist(),
                       "rul_interval": list(log_rul_interval(remaining, radius)) if radius is not None else None})
    return result


def aggregate(rows):
    result = {"rul": summarize(rows, [r["predicted_eol_cycle"] for r in rows]),
              "full_remaining_curve": {}, "fixed_horizons": {}, "intervals": {}}
    ids = sorted({r["cell_id"] for r in rows})
    for name in ("learned_shape", "linear_shape", "persistence"):
        values = [np.mean([r["full_remaining_curve_mape"][name] for r in rows if r["cell_id"] == key and
                           r["full_remaining_curve_mape"][name] is not None]) for key in ids]
        result["full_remaining_curve"][name] = float(np.mean(values))
    for name in ("10", "50", "100", "20pct_remaining"):
        per_cell = [np.mean([r["fixed_horizon_ape_percent"][name] for r in rows if r["cell_id"] == key and
                            r["fixed_horizon_ape_percent"][name] is not None])
                    for key in ids if any(r["cell_id"] == key and r["fixed_horizon_ape_percent"][name] is not None for r in rows)]
        result["fixed_horizons"][name] = {"cell_macro_mape_percent": float(np.mean(per_cell)) if per_cell else None,
                                         "cells": len(per_cell), "origins": sum(r["fixed_horizon_ape_percent"][name] is not None for r in rows)}
    for fraction in (.2, .4, .6, .8):
        scored = [r for r in rows if r["fraction"] == fraction and r["rul_interval"] is not None]
        if scored:
            covered = [r["rul_interval"][0] <= r["actual_eol_cycle"] - r["origin_cycle"] <= r["rul_interval"][1] for r in scored]
            widths = [r["rul_interval"][1] - r["rul_interval"][0] for r in scored]
            result["intervals"][str(int(fraction * 100))] = {"coverage": float(np.mean(covered)),
                "median_width_cycles": float(np.median(widths)), "mean_width_cycles": float(np.mean(widths)), "cells": len(scored)}
    if result["intervals"]:
        result["joint_four_origin_coverage"] = float(np.mean([all(
            r["rul_interval"][0] <= r["actual_eol_cycle"] - r["origin_cycle"] <= r["rul_interval"][1]
            for r in rows if r["cell_id"] == key) for key in ids]))
    result["anchor_projection_count"] = sum(r["anchor_projected_above_80"] for r in rows)
    return result


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    previous = json.loads((HEALTH / "diagnostic.json").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    events = {r["cell_id"]: int(r["exact80_event_cycle"]) for r in audit["by_cell"] if r["exact80_event_cycle"] is not None}
    records = [r for r in load_manifest()["cells"] if r["cell_id"] in events]
    cells = {r["cell_id"]: load_cell(r) for r in records}
    signals = {key: feature_matrix(cell) for key, cell in cells.items()}
    if len(cells) != 43:
        raise ValueError("fixed event cohort changed")
    save_json(run / "config.json", {"run_id": run_id, "seed": SEED, "proper_training_calibration_test_disjoint": True,
        "shapes": list(SHAPES), "calibration_cells_per_fold": CALIBRATION_CELLS, "nominal_joint_coverage": .9,
        "source_result_sha256": source_hash(HEALTH / "diagnostic.json"), "endpoint_audit_sha256": source_hash(AUDIT),
        "script_sha256": source_hash(__file__), "trajectory_sha256": source_hash("battery_smetimes/models/trajectory.py"),
        "claim_boundary": "43 reused MIT Batch-2 event cells; retrospective life-fraction origins; not cross-batch acceptance"})
    all_rows = {"full_training_point": [], "split_calibrated": []}
    fold_results = []
    print("RUN", run.resolve(), flush=True)
    for fold in previous["folds"]:
        number, train_ids, test_ids = fold["fold"], fold["train_cells"], fold["test_cells"]
        folder = run / f"fold{number}"
        folder.mkdir()
        with np.load(HEALTH / f"fold{number}/soh_predictions.npz") as source:
            estimates = {key: source[key].copy() for key in source.files}
        shape, shape_info = shape_fit(cells, events, train_ids, estimates)
        save_json(folder / "full_training_shape.json", shape_info)
        rows = checkpoint_rows(cells, events, test_ids)
        point_index = {(r["cell_id"], r["fraction"]): r for r in previous["oof_predictions"]["signal_plus_estimated_soh_logrul"]}
        remaining = [point_index[(r["cell_id"], r["fraction"])]["predicted_eol_cycle"] - r["origin_cycle"] for r in rows]
        point_rows = score_curves(rows, remaining, cells, estimates, shape)
        save_json(folder / "full_training_point_predictions.json", point_rows)
        all_rows["full_training_point"].extend(point_rows)

        order = np.random.default_rng(SEED + number).permutation(sorted(train_ids)).tolist()
        calibration, proper = sorted(order[:CALIBRATION_CELLS]), sorted(order[CALIBRATION_CELLS:])
        save_json(folder / "split.json", {"proper_training": proper, "calibration": calibration, "test": test_ids})
        estimates, matrices, bundle = nested_fit(cells, signals, events, proper, calibration, test_ids, folder)
        calibrated_shape, shape_info = shape_fit(cells, events, proper, estimates)
        save_json(folder / "calibrated_shape.json", shape_info)
        cal_rows = checkpoint_rows(cells, events, calibration)
        cal_rul = predict_rul(cal_rows, cells, matrices, bundle)
        calibration_residuals = [{"cell_id": r["cell_id"], "fraction": r["fraction"],
                                  "actual_rul": r["actual_eol_cycle"] - r["origin_cycle"],
                                  "predicted_rul": float(p)} for r, p in zip(cal_rows, cal_rul)]
        radius, calibration_info = cellwise_log_rul_radius(calibration_residuals)
        save_json(folder / "interval_calibration.json", {**calibration_info, "radius": radius, "rows": calibration_residuals})
        test_rul = predict_rul(rows, cells, matrices, bundle)
        calibrated_rows = score_curves(rows, test_rul, cells, estimates, calibrated_shape, radius)
        save_json(folder / "split_calibrated_predictions.json", calibrated_rows)
        all_rows["split_calibrated"].extend(calibrated_rows)
        fold_result = {"fold": number, "full_training_shape": shape, "calibrated_shape": calibrated_shape,
                       "full_training_point": aggregate(point_rows), "split_calibrated": aggregate(calibrated_rows)}
        save_json(folder / "result.json", fold_result)
        fold_results.append(fold_result)
        save_json(run / "progress.json", {"completed_folds": len(fold_results), "total_folds": 5, "folds": fold_results})
        print("SAVED fold", number, "shape", shape, calibrated_shape,
              "curve MAPE", fold_result["full_training_point"]["full_remaining_curve"]["learned_shape"],
              "coverage", fold_result["split_calibrated"].get("joint_four_origin_coverage"), flush=True)
    if any(len(rows) != 172 for rows in all_rows.values()):
        raise ValueError("expected all four origins for 43 event cells")
    result = {"run_id": run_id, "status": "development; full remaining observed curves and cell-disjoint interval calibration",
              "folds": fold_results, "aggregate": {name: aggregate(rows) for name, rows in all_rows.items()},
              "predictions": all_rows, "excluded_right_censored_cells": 91,
              "limitations": ["intervals apply to this selected event cohort and retrospective origins",
                               "finite calibration sample and one batch; no new-domain coverage guarantee",
                               "curve endpoint anchored to RUL, so consistency is by construction",
                               "no measured future SOH or test-fit shape at inference"]}
    save_json(run / "result.json", result)
    save_json(OUT / "latest.json", {"run_id": run_id, "run_path": str(run), "result_path": str(run / "result.json")})
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
