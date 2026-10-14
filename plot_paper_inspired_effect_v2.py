"""Plot the paper-inspired development challenger against unchanged baselines."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from battery_v2_features import load_all
from evaluate_paper_inspired_soh_v2 import OUT, fit_gated
from evaluate_relative_soh_features_v2 import baseline_oof, fit_view, predict_view
from paper_inspired_battery_v2 import time_gated_matrix
from train_soh_v2 import outer_splits


DOMAINS = ("NASA", "CALCE", "MIT", "XJTU")
SOH_CV = OUT / "development_cv.json"
RUL_CV = OUT / "rul_development_cv.json"


def metric_plots():
    soh = json.loads(SOH_CV.read_text(encoding="utf-8"))["domains"]
    old = [soh[domain]["summary"]["baseline"]["cell_macro_mape_percent"]
           for domain in DOMAINS]
    new = [soh[domain]["summary"]["gated" if domain == "MIT" else "fusion"]
           ["cell_macro_mape_percent"] if domain in ("MIT", "XJTU") else old[index]
           for index, domain in enumerate(DOMAINS)]
    x = np.arange(len(DOMAINS))
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    for offset, values, label, color in ((-.19, old, "Existing baseline", "#89919a"),
                                          (.19, new, "Chosen candidate", "#087e8b")):
        bars = ax.bar(x + offset, values, .36, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}%", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    ax.axhline(.5, color="#b13c4b", linestyle="--", label="0.5% target")
    ax.set_xticks(x, DOMAINS)
    ax.set_ylabel("Current SOH cell-macro MAPE (%)")
    ax.set_title("SOH: whole-cell / whole-batch development CV")
    ax.set_ylim(0, max(old) + .55)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "soh_baseline_vs_challenger.png", dpi=180)
    plt.close(fig)

    rul = json.loads(RUL_CV.read_text(encoding="utf-8"))["domains"]["XJTU"]
    baseline, candidate = rul["summary"]["baseline"], rul["summary"]["gated_log"]
    baseline_fold_mae = np.mean([row["baseline"]["cell_macro_mae_cycles"]
                                 for row in rul["folds"]])
    challenger_fold_mae = np.mean([row["gated_log"]["cell_macro_mae_cycles"]
                                   for row in rul["folds"]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, values, title, unit in (
        (axes[0], (baseline_fold_mae, challenger_fold_mae),
         "XJTU RUL error across held-out batches", "Equal-batch MAE (cycles)"),
        (axes[1], (baseline["eol_mape_at_50_percent"],
                   candidate["eol_mape_at_50_percent"]),
         "80%-SOH endpoint predicted at cycle 50", "EOL-cycle MAPE (%)")):
        bars = ax.bar(("Existing", "Candidate"), values, color=("#89919a", "#087e8b"), width=.58)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11)
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("RUL: XJTU development CV (not an independent test)")
    fig.savefig(OUT / "xjtu_rul_baseline_vs_challenger.png", dpi=180)
    plt.close(fig)


def soh_curve_plot():
    manifest, cells = load_all()
    comparisons = (("MIT", "batch1", "b1c23", "gated"),
                   ("XJTU", "batch5", "b5_RW_battery-8", "fusion"))
    report = json.loads(SOH_CV.read_text(encoding="utf-8"))["domains"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, (domain, held_out, key, method) in zip(axes, comparisons):
        split = next((training, validation) for name, training, validation
                     in outer_splits(manifest["cells"], domain) if name == held_out)
        training, validation = split
        assert key in validation
        matrices = {item: time_gated_matrix(cells[item]) for item in [*training, key]}
        gate = fit_gated(cells, training, matrices)
        challenger = gate[1].predict(gate[0].transform(matrices[key]))
        if method == "fusion":
            raw = fit_view(cells, training, "raw_only")
            challenger = .5 * (challenger + predict_view(raw, cells[key], "raw_only"))
        baseline = baseline_oof(cells, domain)[key]
        cell = cells[key]
        keep = cell.soh_available
        actual = cell.soh[keep]
        current_mape = 100 * np.mean(abs(challenger[keep] - actual) / actual)
        expected = next(row["mape_percent"] for row in report[domain]["summary"][method]["by_cell"]
                        if row["cell_id"] == key)
        if not np.isclose(current_mape, expected, atol=1e-5):
            raise ValueError(f"recomputed held-out curve does not match CV report: {key}")
        baseline_mape = 100 * np.mean(abs(baseline[keep] - actual) / actual)
        ax.plot(cell.cycle[keep], actual, color="#24282b", linewidth=2.2, label="Measured SOH")
        ax.plot(cell.cycle[keep], baseline[keep], color="#89919a", linewidth=1.8,
                label=f"Existing ({baseline_mape:.2f}%)")
        ax.plot(cell.cycle[keep], challenger[keep], color="#087e8b", linewidth=1.8,
                label=f"Candidate ({current_mape:.2f}%)")
        ax.axhline(.8, color="#b13c4b", linestyle="--", alpha=.8, label="80% SOH")
        ax.set_title(f"{domain} {key} | held out: {held_out}")
        ax.set_xlabel("Source cycle / diagnostic index")
        ax.set_ylabel("Current SOH")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Example held-out SOH traces: observed data available through each cycle")
    fig.savefig(OUT / "soh_heldout_curve_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    metric_plots()
    soh_curve_plot()
    print("Wrote three figures to", OUT)
