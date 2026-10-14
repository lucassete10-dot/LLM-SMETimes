"""Development-only view of XJTU charge-only versus full diagnostic input."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHARGE_A = ROOT / "battery_models_v3/xjtu_charge_stage_a/development_results.json"
CHARGE_B = ROOT / "battery_models_v3/xjtu_charge_stage_b/development_results.json"
FULL_A = ROOT / "battery_models_v3/xjtu_waveform_stage_a/development_results.json"
FULL_B = ROOT / "battery_models_v3/xjtu_stage_b_protocol_holdout/development_results.json"
OUT = ROOT / "battery_models_v3/xjtu_charge_stage_b/charge_availability_gap.png"


def main():
    charge_a = json.loads(CHARGE_A.read_text(encoding="utf-8"))
    charge_b = json.loads(CHARGE_B.read_text(encoding="utf-8"))
    full_a = json.loads(FULL_A.read_text(encoding="utf-8"))
    full_b = json.loads(FULL_B.read_text(encoding="utf-8"))
    selected_a = charge_a["validation_selected_charge_only"]
    selected_b = charge_b["validation_selected_charge_only"]
    panels = (
        (charge_a["models"][selected_a]["test"],
         full_a["models"]["waveform_gru_physics"]["test"]),
        (charge_b["models"][selected_b]["test"],
         full_b["models"]["waveform_gru_physics"]["test"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    groups = ["Whole-cell holdout\n8 cells", "Protocol holdout\n15 Batch-2 cells"]
    x = np.arange(2)
    for index, label, color, offset in (
            (0, "Charge prefix only (validation-selected)", "#d87522", -.18),
            (1, "Charge + discharge waveform GRU", "#257e78", .18)):
        values = [panel[index]["current_cell_macro_mape_percent"] for panel in panels]
        axes[0].bar(x + offset, values, width=.35, label=label, color=color)
        for position, value in zip(x + offset, values):
            axes[0].text(position, value + .04, f"{value:.2f}%",
                         ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x, groups)
    axes[0].set_ylim(0, 2.75)
    axes[0].set_ylabel("Current SOH cell-macro MAPE (%)")
    axes[0].set_title("Held-out development error")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    charge_cell = {row["cell_id"]: row["current_mape_percent"]
                   for row in panels[1][0]["by_cell"]}
    full_cell = {row["cell_id"]: row["current_mape_percent"]
                 for row in panels[1][1]["by_cell"]}
    if set(charge_cell) != set(full_cell):
        raise ValueError("Batch-2 per-cell comparison is not paired")
    paired_delta = np.asarray([charge_cell[key] - full_cell[key]
                               for key in sorted(charge_cell)])
    rng = np.random.default_rng(2026)
    draws = rng.integers(len(paired_delta), size=(20000, len(paired_delta)))
    paired_report = {
        "status": "inspected XJTU Batch-2 development cells; not independent test",
        "metric": "charge-only selected GRU minus full-input waveform GRU current-SOH cell MAPE",
        "mean_difference_percentage_points": float(paired_delta.mean()),
        "cells_charge_only_worse": int(np.sum(paired_delta > 0)),
        "paired_cells": len(paired_delta),
        "cell_bootstrap_95pct_ci_percentage_points":
            [float(value) for value in np.quantile(paired_delta[draws].mean(axis=1),
                                                    (.025, .975))],
    }
    OUT.with_name("charge_availability_gap_audit.json").write_text(
        json.dumps(paired_report, indent=2), encoding="utf-8")
    ids = sorted(charge_cell, key=lambda key: charge_cell[key] - full_cell[key])
    y = np.arange(len(ids))
    axes[1].plot([full_cell[key] for key in ids], y, "o", color="#257e78",
                 label="Charge + discharge GRU")
    axes[1].plot([charge_cell[key] for key in ids], y, "s", color="#d87522",
                 label="Charge-only selected GRU")
    for position, key in enumerate(ids):
        axes[1].plot([full_cell[key], charge_cell[key]], [position, position],
                     color="#b9b9b9", zorder=0)
    axes[1].set_yticks(y, [key.replace("b2_3C_battery-", "cell ") for key in ids])
    axes[1].set_xlabel("Current SOH MAPE (%)")
    axes[1].set_title("Same 15 unseen 3C-charge cells")
    axes[1].grid(axis="x", alpha=.2)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("XJTU input-availability gap: laboratory diagnostic charge prefix vs full charge/discharge",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
