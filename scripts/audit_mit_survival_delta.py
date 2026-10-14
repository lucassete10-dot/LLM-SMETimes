"""Paired-cell bootstrap for MIT RUL challenger deltas (development only)."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "battery_models_v3/mit_exact80_rul_stage_a"


def ape(row):
    actual = row["actual_eol_cycle"] - row["origin_cycle"]
    estimated = max(0, row["predicted_eol_cycle"] - row["origin_cycle"])
    return 100 * abs(estimated - actual) / actual


def paired_difference(first, second, fraction, rng, repeats=20000):
    left = {row["cell_id"]: ape(row) for row in first if row["fraction"] == fraction}
    right = {row["cell_id"]: ape(row) for row in second if row["fraction"] == fraction}
    if set(left) != set(right) or len(left) != 43:
        raise ValueError("paired RUL comparison requires the same 43 cells")
    ids = sorted(left)
    delta = np.asarray([left[key] - right[key] for key in ids])
    draw = rng.integers(len(ids), size=(repeats, len(ids)))
    means = delta[draw].mean(axis=1)
    return {
        "cells": len(ids),
        "mean_delta_mape_points": float(delta.mean()),
        "median_cell_delta_ape_points": float(np.median(delta)),
        "improved_cells": int(np.sum(delta < 0)),
        "bootstrap_95pct_mean_delta_ci": [float(x) for x in np.quantile(means, [.025, .975])],
        "bootstrap_probability_mean_delta_below_zero": float(np.mean(means < 0)),
        "mean_delta_excluding_shortest_life_cell": float(np.mean([
            left[key] - right[key] for key in ids if key != "b2c1"])),
    }


def main():
    tree = json.loads((FOLDER / "five_fold_cell_oof.json").read_text(encoding="utf-8"))
    survival = json.loads((FOLDER / "censored_survival_five_fold_cv.json")
                          .read_text(encoding="utf-8"))
    rng = np.random.default_rng(2026)
    comparisons = {
        "aft_events_only_minus_extra_trees": (
            survival["oof_predictions"]["aft_events_only"],
            tree["oof_predictions"]["extra_trees"]),
        "aft_with_censored_minus_aft_events_only": (
            survival["oof_predictions"]["aft_with_censored"],
            survival["oof_predictions"]["aft_events_only"]),
    }
    report = {
        "status": "development paired-cell bootstrap; not independent test",
        "bootstrap_resamples": 20000,
        "negative_delta_means_first_named_model_has_lower_RUL_MAPE": True,
        "comparisons": {name: {
            str(int(100 * fraction)): paired_difference(first, second, fraction, rng)
            for fraction in (.2, .4, .6, .8)}
            for name, (first, second) in comparisons.items()},
    }
    target = FOLDER / "survival_paired_bootstrap.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {fraction: {
        "mean_delta": round(row["mean_delta_mape_points"], 1),
        "ci": [round(value, 1) for value in row["bootstrap_95pct_mean_delta_ci"]]}
        for fraction, row in result.items()}
        for name, result in report["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()
