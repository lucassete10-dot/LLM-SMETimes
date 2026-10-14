"""Summarize the frozen three-seed timing ablation and render all test cells."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.train_xjtu_cycle_time import MODES, OUT, SEEDS, SOURCES, summarize


COLORS = {"baseline": "#637081", "elapsed": "#2474b5", "elapsed_rate": "#d27816"}
LABELS = {"baseline": "Original input", "elapsed": "+ elapsed cycles",
          "elapsed_rate": "+ elapsed cycles & rates"}


def load_predictions(root, stage, backbone, mode, seed):
    with np.load(root / "runs" / f"{stage}_{backbone}_{mode}_{seed}" / "test.npz") as data:
        return {key: data[key].copy() for key in data.files}


def aligned(first, second):
    for key in ("cell_id", "cycle", "target", "future_target", "horizons"):
        same = np.array_equal(first[key], second[key], equal_nan=True) if \
            first[key].dtype.kind in "fc" else np.array_equal(first[key], second[key])
        if not same:
            raise ValueError(f"unaligned prediction array: {key}")


def paired_diagnostics(root, stage, backbone, mode):
    comparisons, bias_rows = [], []
    for seed in SEEDS:
        base = load_predictions(root, stage, backbone, "baseline", seed)
        new = load_predictions(root, stage, backbone, mode, seed)
        aligned(base, new)
        for key in sorted(set(base["cell_id"])):
            keep = base["cell_id"] == key
            truth = base["target"][keep]
            old_error = 100 * np.abs(base["prediction"][keep] - truth) / truth
            residual = new["prediction"][keep] - truth
            new_error = 100 * np.abs(residual) / truth
            comparisons.append({"seed": seed, "cell_id": key,
                                "current_mape_delta_pp": float(np.mean(new_error) - np.mean(old_error))})
            bias_rows.append({"seed": seed, "cell_id": key,
                              "mape_percent": float(new_error.mean()),
                              "bias_soh_pp": float(residual.mean() * 100),
                              "oracle_centered_mape_percent": float(np.mean(
                                  100 * np.abs(residual - residual.mean()) / truth))})
    per_cell = np.array([np.mean([row["current_mape_delta_pp"] for row in comparisons
                                  if row["cell_id"] == key])
                         for key in sorted({row["cell_id"] for row in comparisons})])
    rng = np.random.default_rng(20928)
    bootstrap = rng.choice(per_cell, (5000, len(per_cell)), replace=True).mean(axis=1)
    return {
        "paired_cell_macro_mape_delta_pp": float(per_cell.mean()),
        "cells_improved_after_seed_average": int(np.sum(per_cell < 0)),
        "cells": len(per_cell),
        "within_cohort_cell_bootstrap95_delta_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
        "seed_macro_mape_deltas_pp": {str(seed): float(np.mean([
            row["current_mape_delta_pp"] for row in comparisons if row["seed"] == seed])) for seed in SEEDS},
        "oracle_centering_is_diagnostic_only": "uses test-cell mean truth; not an available inference/calibration method",
        "raw_mape_mean": float(np.mean([r["mape_percent"] for r in bias_rows])),
        "absolute_bias_soh_pp_mean": float(np.mean([abs(r["bias_soh_pp"]) for r in bias_rows])),
        "oracle_centered_mape_mean": float(np.mean([r["oracle_centered_mape_percent"] for r in bias_rows])),
        "by_cell_seed": bias_rows,
    }


def curve_figure(root, stage, backbone):
    # Predeclared seed, never chosen for its test error.
    data = {mode: load_predictions(root, stage, backbone, mode, 2026) for mode in MODES}
    base = data["baseline"]
    for value in data.values():
        aligned(base, value)
    keys = sorted(set(base["cell_id"]))
    columns = 2 if stage == "A" else 3
    rows = (len(keys) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5.5, rows * 2.8), squeeze=False)
    for axis, key in zip(axes.flat, keys):
        mask = base["cell_id"] == key
        axis.plot(base["cycle"][mask], base["target"][mask], color="#202020", lw=1.7, label="Measured SOH")
        for mode in MODES:
            axis.plot(base["cycle"][mask], data[mode]["prediction"][mask],
                      color=COLORS[mode], lw=1.1, label=LABELS[mode])
        axis.set_title(str(key), fontsize=9)
        axis.set_xlabel("Source cycle", fontsize=9)
        axis.set_ylabel("SOH", fontsize=9)
        axis.grid(alpha=.2)
    for axis in axes.flat[len(keys):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, .972), frameon=False)
    title = "unseen-cell" if stage == "A" else "held-out protocol"
    fig.suptitle(f"XJTU {title} development curves | {backbone.upper()} | fixed seed 2026", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .935))
    fig.savefig(root / f"stage_{stage.lower()}_current_curves.png", dpi=155)
    plt.close(fig)


def overview_figure(root, summary):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    x = np.arange(2)
    for row, stage in enumerate(SOURCES):
        candidates = summary["splits"][stage]["candidates"]
        for column, task in enumerate(("current", "future100")):
            axis = axes[row, column]
            for i, mode in enumerate(MODES):
                values, errors = [], []
                for backbone in ("gru", "tcn"):
                    candidate = candidates[f"{backbone}_{mode}"]["test"]
                    values.append(candidate["current_mape_mean"] if task == "current"
                                  else candidate["future_mape_mean"]["100"])
                    errors.append(candidate["current_mape_sd"] if task == "current" else 0)
                bars = axis.bar(x + (i - 1) * .24, values, width=.23,
                                yerr=errors if task == "current" else None,
                                capsize=3, color=COLORS[mode], label=LABELS[mode])
                axis.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
            axis.set_xticks(x, ["GRU", "TCN"])
            axis.set_ylabel("Cell-macro MAPE (%)")
            axis.set_title(f"Stage {stage}: " + ("current SOH" if task == "current" else "rolling +100-cycle SOH"))
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
            axis.set_ylim(bottom=0, top=axis.get_ylim()[1] * 1.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.suptitle("Timing-input ablation: three-seed means on fixed development cells", fontsize=14)
    fig.text(.5, .015, "A: within-condition unseen cells | B: held-out Batch 2 | Error bars: training-seed SD, not confidence intervals",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .89))
    fig.savefig(root / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUT)
    root = parser.parse_args().input
    summary = summarize(root)
    if summary["completed_runs"] != 36 or any(
            row["selected_on_validation"] is None for row in summary["splits"].values()):
        raise ValueError("all 36 preregistered runs must finish before final reporting")
    diagnostics = {}
    lines = ["# XJTU real-cycle timing ablation", "",
             "Three predeclared seeds (2026/2027/2028), frozen whole-cell A/B splits,",
             "identical training budget, preprocessing and model parameters within each backbone.",
             "All arms retain the original signal features. A zero-initialized 48-to-64",
             "projection gets zero context (baseline), actual gaps/relative cycles (elapsed),",
             "or these plus trailing feature slopes per source cycle (elapsed_rate).",
             "The baseline initial function exactly matches the original GRU/TCN.", "",
             "Stage A uses 39/8/8 train/validation/test cells. Stage B trains on",
             "Batches 1/4/5/6, validates on Batch 3 and tests on Batch 2.",
             "These are inspected development cohorts; seed variation is not new-cell evidence.", "",
             "| Split | Model/input | Current test MAPE mean ± SD | +10 | +50 | +100 | Validation score |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage, block in summary["splits"].items():
        for name, candidate in block["candidates"].items():
            test = candidate["test"]
            future = test["future_mape_mean"]
            lines.append(f"| {stage} | {name} | {test['current_mape_mean']:.3f}% ± {test['current_mape_sd']:.3f} | "
                         f"{future['10']:.3f}% | {future['50']:.3f}% | {future['100']:.3f}% | "
                         f"{candidate['validation']['selection_score_mean']:.3f} |")
        for backbone in ("gru", "tcn"):
            for mode in MODES:
                diagnostics[f"{stage}_{backbone}_{mode}"] = paired_diagnostics(root, stage, backbone, mode)
        selected = block["selected_on_validation"]
        curve_figure(root, stage, selected.split("_", 1)[0])
    lines += ["", "## Validation selection", "",
              "Selection uses mean validation current MAPE + 0.2 × mean future MAPE across all three seeds.",
              "No seed or architecture is selected from test results.", ""]
    for stage, block in summary["splits"].items():
        lines.append(f"- Stage {stage}: `{block['selected_on_validation']}`.")
    lines += ["", "## Interpretation boundaries", "",
              "The +10/+50/+100 scores cover exact available future diagnostic labels,",
              "not an entire remaining-life forecast. XJTU has no confirmed 80%-of-own-reference",
              "events under the frozen label contract, so this experiment does not validate RUL.",
              "`paired_diagnostics.json` compares identical cells/seeds and includes per-cell bias.",
              "Its oracle-centered residual score subtracts the full test-cell mean error solely",
              "to diagnose offset versus shape; it is not a realizable online calibration result.",
              "Cell-bootstrap intervals are conditional on these cohorts and average over seeds.", "",
              "## Reproduction and artifacts", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -m scripts.train_xjtu_cycle_time",
              ".\\.venv\\Scripts\\python.exe -m scripts.report_xjtu_cycle_time", "```", "",
              "Each `runs/<split>_<backbone>_<arm>_<seed>/` has a reloadable `model.pt`,",
              "`result.json` with validation-selected epoch/training history/per-cell scores,",
              "and `validation.npz`/`test.npz` with every prediction and aligned target.",
              "`experiment_plan.json` fingerprints the code and in-memory data before fitting.", "",
              "![Three-seed comparison](comparison.png)", "",
              "![Stage A, all held-out cells](stage_a_current_curves.png)", "",
              "![Stage B, all held-out cells](stage_b_current_curves.png)", ""]
    (root / "paired_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    overview_figure(root, summary)
    print(root / "README.md")
    print(json.dumps({key: row["selected_on_validation"] for key, row in summary["splits"].items()}))


if __name__ == "__main__":
    main()
