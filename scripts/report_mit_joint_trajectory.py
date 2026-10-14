"""Verify saved joint models, export a label-free example and plot the run."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from battery_smetimes.models.trajectory import anchored_soh
from battery_v2_features import load_cell, load_manifest
from scripts.predict_mit_joint_trajectory import predict_history
from scripts.train_mit_joint_trajectory import OUT, save_json


def verify_and_export(run, report, cells):
    checks = []
    example_saved = False
    for fold in report["folds"]:
        number = fold["fold"]
        folder = run / f"fold{number}"
        split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
        groups = [set(split[name]) for name in ("proper_training", "calibration", "test")]
        if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("fold groups overlap")
        for filename in ("soh_predictor.joblib", "rul_predictor.joblib"):
            if set(joblib.load(folder / filename)["train_cells"]) != groups[0]:
                raise ValueError("saved weights do not match proper training group")
        rows = [r for r in report["predictions"]["split_calibrated"] if r["cell_id"] in groups[2]]
        maximum = 0.
        for row in rows:
            cell = cells[row["cell_id"]]
            position = np.searchsorted(cell.cycle, row["origin_cycle"]) + 1
            forecast = predict_history(cell.cycle[:position], cell.matrix("signal_history")[:position],
                                       cell.reference_count, folder)
            difference = abs(forecast["rul_cycles"] - row["predicted_rul"])
            maximum = max(maximum, difference)
            if difference > 1e-7 or abs(forecast["estimated_current_soh"] - row["estimated_current_soh"]) > 1e-7:
                raise ValueError("saved-model prefix-only inference differs from recorded evaluation")
            if not example_saved:
                np.savez_compressed(run / "example_input.npz", cycle=cell.cycle[:position],
                    signal_history=cell.matrix("signal_history")[:position], reference_count=cell.reference_count)
                save_json(run / "example_prediction.json", forecast)
                save_json(run / "example_provenance.json", {"cell_id": row["cell_id"], "fold": number,
                    "origin_cycle": row["origin_cycle"], "input_fields": ["cycle", "signal_history", "reference_count"],
                    "prediction_uses_no_capacity_soh_or_eol_labels": True})
                example_saved = True
        checks.append({"fold": number, "verified_origins": len(rows),
                       "maximum_rul_reload_difference_cycles": maximum,
                       "train_calibration_test_disjoint": True})
    save_json(run / "reload_verification.json", {"passed": True, "origins": sum(r["verified_origins"] for r in checks), "folds": checks})


def plot_curves(run, report, cells):
    rows = report["predictions"]["full_training_point"]
    events = {r["cell_id"]: r["actual_eol_cycle"] for r in rows}
    keys = sorted(events, key=events.get)
    selected = [keys[0], keys[len(keys) // 2], keys[-1]]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for i, key in enumerate(selected):
        cell = cells[key]
        for j, fraction in enumerate((.2, .4, .6, .8)):
            axis = axes[i, j]
            row = next(r for r in rows if r["cell_id"] == key and r["fraction"] == fraction)
            origin, event = row["origin_cycle"], row["actual_eol_cycle"]
            past = cell.soh_available & (cell.cycle <= origin)
            axis.plot(cell.cycle[past], cell.soh[past], color="#202020", lw=1.6, label="Observed history")
            axis.plot(row["future_cycle"], row["future_true_soh"], color="#777777", lw=1.5, label="Measured future")
            end = max(event, row["predicted_eol_cycle"]) * 1.05
            h = np.linspace(0, end - origin, 250)
            prediction = anchored_soh(row["estimated_current_soh"], row["predicted_rul"], h, row["shape"])
            axis.plot(origin + h, prediction, color="#177c9b", lw=1.8, label="Predicted future")
            axis.axvline(origin, color="#bc7926", linestyle=":", lw=1)
            axis.axhline(.8, color="#ad3745", linestyle="--", lw=.8)
            axis.set_title(f"{key} | observed {fraction:.0%}\nRUL true {event-origin}, pred {row['predicted_rul']:.0f}", fontsize=9)
            axis.set_xlabel("Source cycle", fontsize=9)
            axis.set_ylabel("SOH", fontsize=9)
            axis.set_ylim(.72, 1.04)
            axis.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .965), ncol=3, frameon=False)
    fig.suptitle("MIT full remaining SOH curves: shortest / median / longest observed life", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .925))
    fig.savefig(run / "full_curve_examples.png", dpi=160)
    plt.close(fig)


def plot_intervals(run, report):
    metrics = report["aggregate"]["split_calibrated"]
    keys = ("20", "40", "60", "80")
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    coverage = [100 * metrics["intervals"][key]["coverage"] for key in keys]
    bars = axes[0].bar(x, coverage, color="#327a98")
    axes[0].bar_label(bars, fmt="%.1f%%", padding=4)
    axes[0].axhline(90, color="#b74942", linestyle="--", label="Nominal 90%")
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_title(f"All-four-origin coverage: {metrics['joint_four_origin_coverage']:.1%}")
    axes[0].legend(frameon=False)
    widths = [metrics["intervals"][key]["median_width_cycles"] for key in keys]
    errors = [metrics["rul"][key]["rul_mae_cycles"] for key in keys]
    axes[1].bar(x - .18, widths, width=.35, label="Median interval width", color="#9ab8c3")
    axes[1].bar(x + .18, errors, width=.35, label="Point-prediction MAE", color="#db9139")
    axes[1].set_ylabel("Cycles")
    axes[1].set_title("Coverage comes with broad intervals")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x, [key + "%" for key in keys])
        axis.set_xlabel("Observed fraction of event life")
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("MIT: disjoint training / calibration / test cells", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(run / "rul_interval_coverage.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    run = args.run or Path(json.loads((OUT / "latest.json").read_text(encoding="utf-8"))["run_path"])
    report = json.loads((run / "result.json").read_text(encoding="utf-8"))
    ids = {r["cell_id"] for r in report["predictions"]["full_training_point"]}
    cells = {r["cell_id"]: load_cell(r) for r in load_manifest()["cells"] if r["cell_id"] in ids}
    verify_and_export(run, report, cells)
    plot_curves(run, report, cells)
    plot_intervals(run, report)
    point, calibrated = (report["aggregate"][name] for name in ("full_training_point", "split_calibrated"))
    lines = ["# MIT full-trajectory and RUL-interval development experiment", "",
             f"Run: `{report['run_id']}`. All run artifacts are retained in this timestamped folder.", "",
             "The same 43 observed-event MIT Batch-2 cells and five outer cell folds are reused.",
             "A shared shifted-power curve is fitted on training-cell capacity histories and",
             "anchored at the model-estimated current SOH and predicted 80%-SOH endpoint.",
             "This produces a consistent SOH/RUL pair by construction. Its shape is supervised",
             "only on source-cell future labels; query future capacity/EOL is never an input.", "",
             "## Full remaining observed curve", "",
             "Scores average over cells, then equally over their four retrospective origins.",
             "Each curve is scored on all available future capacity diagnostics through actual 80% EOL.", "",
             "| Route | Learned shape | Linear shape, same endpoint | Current-SOH persistence |",
             "| --- | ---: | ---: | ---: |"]
    for name, metric in (("All outer training cells", point), ("Separate 10-cell calibration holdout", calibrated)):
        values = metric["full_remaining_curve"]
        lines.append(f"| {name} | {values['learned_shape']:.3f}% | {values['linear_shape']:.3f}% | {values['persistence']:.3f}% |")
    lines += ["", "| Point-model forecast horizon | Cell-macro MAPE | Cells / origins |",
              "| --- | ---: | ---: |"]
    for key, metric in point["fixed_horizons"].items():
        lines.append(f"| {key} | {metric['cell_macro_mape_percent']:.3f}% | {metric['cells']} / {metric['origins']} |")
    lines += ["", "## RUL interval calibration", "",
              "In each outer fold, 10 cells are set aside for calibration and never fit either",
              "the SOH estimator, RUL model, imputer or shape. Inner whole-cell SOH predictions",
              "supply the proper-training RUL inputs. Calibration uses each cell's maximum",
              "absolute log-RUL residual over its four checkpoints; the finite-sample 90%",
              "quantile is the tenth of ten cell scores. The four checkpoints are retrospective",
              "fractions of true lifetime, so prospective-origin calibration is not established.", "",
              "| Observed life | Point RUL MAPE | Calibrated-model RUL MAPE | Interval coverage | Median interval width |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for key in ("20", "40", "60", "80"):
        interval = calibrated["intervals"][key]
        lines.append(f"| {key}% | {point['rul'][key]['rul_mape_percent']:.2f}% | "
                     f"{calibrated['rul'][key]['rul_mape_percent']:.2f}% | {interval['coverage']:.1%} | {interval['median_width_cycles']:.1f} cycles |")
    lines += ["", f"All-four-origin empirical coverage is **{calibrated['joint_four_origin_coverage']:.1%}**,",
              "below the nominal 90%. Per-checkpoint coverage must not be substituted for this",
              "joint target. Intervals are broad and do not yet demonstrate precise uncertainty",
              "for an individual battery. These finite development cohorts do not support",
              "new-batch/new-chemistry coverage guarantees. The 91 right-censored MIT cells",
              "are preserved in the source data and are not assigned invented point events here.", "",
              "The selected curve shape is 3.0 in every fold, the boundary of the predefined grid.",
              "This simple family may underrepresent the knee and regeneration; curve agreement",
              "does not independently validate the direct RUL head. No cell needed projection",
              "of a below-80% estimated anchor in this run.", "",
              "## Saved weights and independent reload check", "",
              "`reload_verification.json` checks disjoint cell identities and replays all 172",
              "test origins using saved weights and **prefix-only, label-free** input features.",
              "The standalone inference command accepts no capacity/SOH/EOL label:", "", "```powershell",
              f".\\.venv\\Scripts\\python.exe -m scripts.predict_mit_joint_trajectory --features {run / 'example_input.npz'} --fold-dir {run / 'fold1'} --output {run / 'prediction_replay.json'}",
              "```", "",
              "The saved example prediction and provenance are already present. Choose a new",
              "output filename for each replay; the inference command will not overwrite one.",
              "Each fold has SOH inner/outer weights, RUL weights, shape, calibration residuals,",
              "full true/predicted trajectories, and metrics. `config.json` records seed and hashes.", "",
              "![Full curve examples](full_curve_examples.png)", "",
              "![Interval coverage and width](rul_interval_coverage.png)", ""]
    (run / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(run.resolve())
    print("Verified 172 saved-model prefix predictions; figures and report saved.")


if __name__ == "__main__":
    main()
