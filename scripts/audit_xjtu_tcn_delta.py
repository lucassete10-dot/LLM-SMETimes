"""Paired-cell development uncertainty for XJTU waveform TCN versus GRU."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TCN = ROOT / "battery_models_v3/xjtu_tcn_stage_a/development_results.json"
GRU = ROOT / "battery_models_v3/xjtu_waveform_stage_a/development_results.json"


def bootstrap_difference(tcn, gru, metric, rng):
    left = {row["cell_id"]: metric(row) for row in tcn}
    right = {row["cell_id"]: metric(row) for row in gru}
    ids = sorted(key for key in left if left[key] is not None and
                 right.get(key) is not None)
    delta = np.asarray([left[key] - right[key] for key in ids])
    draws = rng.integers(len(ids), size=(20000, len(ids)))
    interval = np.quantile(delta[draws].mean(axis=1), (.025, .975))
    return {
        "scored_cells": ids,
        "mean_tcn_minus_gru_mape_points": float(delta.mean()),
        "cells_tcn_worse": int(np.sum(delta > 0)),
        "paired_bootstrap_95pct_ci": interval.tolist(),
    }


def main():
    tcn = json.loads(TCN.read_text(encoding="utf-8"))
    gru = json.loads(GRU.read_text(encoding="utf-8"))
    rng = np.random.default_rng(2026)
    result = {"status": "inspected development holdout, not independent test",
              "positive_delta_means_TCN_worse": True,
              "splits": {}}
    for part in ("validation", "test"):
        left = tcn["models"]["waveform_tcn_physics"][part]["by_cell"]
        right = gru["models"]["waveform_gru_physics"][part]["by_cell"]
        result["splits"][part] = {
            "current_soh": bootstrap_difference(
                left, right, lambda row: row["current_mape_percent"], rng),
            "future_100_cycles": bootstrap_difference(
                left, right, lambda row: row["future"]["100"]["mape_percent"], rng),
        }
    target = TCN.with_name("paired_tcn_gru_delta.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({part: {name: (round(row["mean_tcn_minus_gru_mape_points"], 3),
                                    len(row["scored_cells"]))
                             for name, row in metrics.items()}
                      for part, metrics in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
