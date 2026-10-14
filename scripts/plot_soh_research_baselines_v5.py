"""Evidence-linked completed-baseline figure; no smoothing or cell exclusions."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
import numpy as np

from scripts.prepare_same_type_forecast import save_json, sha


COLORS = {"ridge": "#868D97", "extra_trees": "#386A93", "truth": "#202B34", "target": "#A75151"}
LABELS = {"ridge": "Ridge", "extra_trees": "ExtraTrees"}
DATASETS = ("MIT", "XJTU", "NASA")


def load_curves(run, row, cell, origin, fingerprints):
    own = next(item for item in row["by_cell_training_seed_variation"] if item["cell_id"] == cell)
    predictions, reference_cycle, reference_truth = [], None, None
    for seed in row["training_seeds"]:
        path = run / own["fold"] / str(seed) / row["model"] / "test/predictions.npz"
        fingerprints[str(path.resolve())] = sha(path)
        with np.load(path) as arrays:
            prefix = f"{cell}__{origin}__"
            cycle, truth, prediction = [arrays[prefix + name] for name in ("cycle", "truth", "prediction")]
            if not np.isfinite(prediction).all() or not np.all(np.diff(cycle) > 0):
                raise ValueError("nonfinite prediction or nonmonotone source-cycle grid")
            if reference_cycle is None:
                reference_cycle, reference_truth = cycle, truth
            elif not np.array_equal(cycle, reference_cycle) or not np.array_equal(truth, reference_truth):
                raise ValueError("seed comparison uses different observed targets")
            predictions.append(prediction)
    prediction = np.asarray(predictions)
    return {"cycle": reference_cycle, "truth": reference_truth, "mean": prediction.mean(0),
            "sd": prediction.std(0, ddof=1), "seeds": row["training_seeds"]}


def qa_export(fig, prefix, qa, alignment):
    preflight = subprocess.run([sys.executable, str(qa / "validate_figure.py"), __file__, "--json"],
                               capture_output=True, text=True)
    Path(str(prefix) + ".source-preflight.json").write_text(preflight.stdout + preflight.stderr, encoding="utf-8")
    if preflight.returncode:
        raise RuntimeError("figure source preflight failed; inspect the saved audit")
    fig.canvas.draw()
    alignment.require_matplotlib_panel_alignment(fig, json_out=str(prefix) + ".alignment.json",
        overlay_svg=str(prefix) + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        strict=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(str(prefix) + "." + suffix)
    fig.savefig(str(prefix) + ".png", dpi=400)
    # Figure geometry is authoritative in the PDF; overlays are QA only.
    commands = [([sys.executable, str(qa / "audit_pdf_text.py"), str(prefix) + ".pdf", "--min-pt", "5", "--json"], ".text-audit.json"),
        ([sys.executable, str(qa / "audit_figure_collisions.py"), str(prefix) + ".pdf",
          "--json-out", str(prefix) + ".collision-audit.json", "--overlay-pdf", str(prefix) + ".collision-audit.pdf"], ".collision-audit.log")]
    for command, log in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        Path(str(prefix) + log).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("rendered figure QA failed; inspect the saved audit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--qa-scripts", type=Path, required=True)
    args = parser.parse_args()
    primary = json.loads(args.primary_report.read_text())["primary"]
    rows = {(row["dataset"], row["model"]): row for row in primary
            if row["complete_model_dataset_primary"] and row["model"] in LABELS}
    if set(rows) != {(dataset, model) for dataset in DATASETS for model in LABELS}:
        raise ValueError("completed baseline coverage is required for all three cohorts")
    if len({row["run_id"] for row in rows.values()}) != 1:
        raise ValueError("baseline rows must use one immutable completed experiment")
    run = Path("experiments/soh_research_v5/runs") / next(iter(rows.values()))["run_id"]
    if not json.loads((run / "progress.json").read_text())["complete"]:
        raise ValueError("plot only the completed baseline run")
    qa = args.qa_scripts.resolve()
    spec = importlib.util.spec_from_file_location("audit_panel_alignment", qa / "audit_panel_alignment.py")
    alignment = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = alignment
    spec.loader.exec_module(alignment)
    folder = Path("results/soh_research_v5_figures") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, folder / "plot_source.py")
    contract = {"question": "How reliably do completed baselines predict unseen-cell future SOH, and where do they fail?",
        "claim": "Early-signal ExtraTrees has low cohort-average future error on MIT/XJTU, but individual failures remain and the four-cell NASA cohort misses the target.",
        "archetype": "asymmetric quantitative grid",
        "panels": {"a": "population error and target attainment; all eligible cells",
                   "b": "paired within-cell baseline contrast, not overlapping marginal rankings",
                   "c": "worst NASA ExtraTrees cell selected explicitly as a failure boundary"},
        "units": "one point per battery after averaging five training seeds; cycles and seeds are not extra batteries",
        "uncertainty": "summary markers and forecast bands use training-seed SD; no confidence intervals or significance tests",
        "exclusions": "none of the completed baseline's eligible batteries or future records",
        "selection": "c is the highest NASA ExtraTrees seed-averaged MAPE, not a representative or tuning cell",
        "dimensions_mm": [183, 118], "backend": "Python", "minimum_glyph_pt": 5,
        "transformations": "SOH 0-1; no smoothing, interpolation, clipping, EOL conditioning or lifetime normalization"}
    save_json(folder / "figure_contract.json", contract)
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65})
    fig = plt.figure(figsize=(183 / 25.4, 118 / 25.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], left=.09, right=.975, top=.87,
                           bottom=.21, wspace=.36, hspace=.67)
    a, b, c = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    plot_data, paired_data, fingerprints = [], [], {}
    for index, dataset in enumerate(DATASETS):
        members = {}
        for model in LABELS:
            row = rows[(dataset, model)]
            members[model] = {item["cell_id"]: item["mape_percent"]["mean"] for item in row["by_cell_training_seed_variation"]}
            values = [members[model][cell] for cell in sorted(members[model])]
            jitter = np.linspace(-.055, .055, len(values))
            offset, marker = (-.15, "o") if model == "ridge" else (.15, "D")
            a.scatter(index + offset + jitter, values, s=9, marker=marker, color=COLORS[model], alpha=.45, linewidths=0)
            statistic = row["metrics"]["mape_percent"]["seed_macro_mean_and_sd"]
            a.errorbar(index + offset, statistic["mean"], yerr=statistic["sd"], fmt=marker,
                       color=COLORS[model], markersize=4, capsize=3, linewidth=1.1)
            plot_data.append({"dataset": dataset, "model": model, "cells": members[model], "seed_macro": statistic})
        if set(members["ridge"]) != set(members["extra_trees"]):
            raise ValueError("paired baseline battery sets differ")
        differences = {cell: members["ridge"][cell] - members["extra_trees"][cell] for cell in sorted(members["ridge"])}
        b.scatter(index + np.linspace(-.09, .09, len(differences)), list(differences.values()),
                  s=9, color=COLORS["extra_trees"], alpha=.5, linewidths=0)
        seeds = np.asarray(rows[(dataset, "ridge")]["metrics"]["mape_percent"]["seed_macro_values"]) - np.asarray(rows[(dataset, "extra_trees")]["metrics"]["mape_percent"]["seed_macro_values"])
        b.errorbar(index, seeds.mean(), yerr=seeds.std(ddof=1), fmt="D", color=COLORS["extra_trees"], capsize=3, markersize=4)
        paired_data.append({"dataset": dataset, "cell_paired_difference_pp": differences, "paired_seed_macro_values": seeds.tolist()})
    category_labels = [f'{dataset}{"*" if dataset == "NASA" else ""}\n{rows[(dataset, "ridge")]["expected_cells"]} cells' for dataset in DATASETS]
    for axis in (a, b):
        axis.set_xticks(range(3), category_labels)
        axis.set_xlim(-.45, 2.45)
        axis.grid(axis="y", alpha=.12)
    a.axhline(6, color=COLORS["target"], linestyle="--", linewidth=.8)
    a.set(ylim=(0, None), ylabel="Cell-mean MAPE (%)", title="Future error at source N = 100")
    b.axhline(0, color=COLORS["ridge"], linestyle=":", linewidth=.8)
    b.set(ylabel="Ridge − ExtraTrees (pp)", title="Paired error reduction")
    worst = rows[("NASA", "extra_trees")]["worst_cell"]["cell_id"]
    curve_data = {}
    for model in LABELS:
        curves = load_curves(run, rows[("NASA", model)], worst, 100, fingerprints)
        curve_data[model] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in curves.items()}
        c.plot(curves["cycle"], curves["mean"], color=COLORS[model], linewidth=1.1)
        c.fill_between(curves["cycle"], curves["mean"] - curves["sd"], curves["mean"] + curves["sd"], color=COLORS[model], alpha=.18, linewidth=0)
    c.plot(curves["cycle"], curves["truth"], color=COLORS["truth"], linewidth=1.1)
    c.set(xlabel="Future source cycle", ylabel="SOH", title=f"NASA {worst}: worst ET case")
    for axis, label in zip((a, b, c), "abc"):
        transform = axis.transAxes + ScaledTranslation(-14 / 72, 6 / 72, fig.dpi_scale_trans)
        axis.text(0, 1, label, transform=transform, fontsize=8, fontweight="bold", va="bottom")
    fig.suptitle("Unseen-cell future SOH: completed baselines", y=.98, fontsize=9)
    handles = [Line2D([], [], color=COLORS[model], marker="o" if model == "ridge" else "D", markersize=3, label=LABELS[model]) for model in LABELS]
    handles.extend([Line2D([], [], color=COLORS["truth"], label="Observed SOH"),
                    Line2D([], [], color=COLORS["target"], linestyle="--", label="6% MAPE target")])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .075), ncol=4, frameon=False)
    fig.text(.52, .058, "Five seeds; points are batteries; summary bars / bands are training-seed SD.", ha="center", fontsize=6)
    fig.text(.52, .026, "*NASA: physically unverified four-cell cohort; supplementary development validation.", ha="center", fontsize=6)
    save_json(folder / "source_data.json", {"primary_report_sha256": sha(args.primary_report),
        "plot_source_sha256": sha(Path(__file__)), "population": plot_data, "paired": paired_data,
        "boundary_cell": worst, "boundary_curves": curve_data, "prediction_sources_sha256": fingerprints,
        "qa_helpers_sha256": {name: sha(qa / name) for name in ("validate_figure.py", "audit_panel_alignment.py", "audit_pdf_text.py", "audit_figure_collisions.py")}})
    caption = ("Fig. 1 | Completed baseline future-SOH evaluation. a Cell-level all-observed-future MAPE from source cycle 100; small marks are five-seed-averaged battery errors, large marks are cell-macro means ± training-seed SD. "
        "b Ridge minus ExtraTrees MAPE on the same batteries; positive values favor ExtraTrees. c NASA " + worst + ", explicitly selected as its worst ExtraTrees cell; lines are seed means and bands one training-seed SD. "
        "MIT n=134 and XJTU n=55 batteries use grouped five-fold evaluation with inner validation; NASA n=4 uses leave-one-cell-out and is a physically unverified supplementary cohort. "
        "All models use only the first 100 source cycles of V/I/T-derived inputs and known conditions, without capacity/SOH history or future signals. SOH uses the audited initial-comparable-capacity reference. "
        "Ridge is deterministic; repeated seeds do not add independent batteries. No confidence intervals, hypothesis tests or smoothing are applied. Data are public development validation, not blind validation. Source data and source fingerprints are supplied with the figure.")
    (folder / "caption.md").write_text(caption + "\n", encoding="utf-8")
    qa_export(fig, folder / "baseline_evidence", qa, alignment)
    plt.close(fig)
    save_json(Path("results/latest_soh_research_v5_figure.json"), {"folder": str(folder.resolve())})
    print(json.dumps({"folder": str(folder.resolve()), "status": "export and automated geometry QA completed; visual review still required"}, indent=2))


if __name__ == "__main__":
    main()
