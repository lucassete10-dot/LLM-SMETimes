"""Compare exact-80 RUL targets and measured/predicted health input fairly."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.diagnose_mit_health_rul import OUT as SOURCE
from scripts.diagnose_mit_estimated_health_rul import OUT


def ape_index(rows):
    result = {}
    for row in rows:
        actual = row["actual_eol_cycle"] - row["origin_cycle"]
        predicted = max(0, row["predicted_eol_cycle"] - row["origin_cycle"])
        result[(row["cell_id"], row["fraction"])] = 100 * abs(predicted - actual) / actual
    return result


def main():
    original = json.loads((SOURCE / "diagnostic.json").read_text(encoding="utf-8"))
    nested = json.loads((OUT / "diagnostic.json").read_text(encoding="utf-8"))
    metrics = {**original["aggregate"], **nested["aggregate"]}
    predictions = {**original["oof_predictions"], **nested["oof_predictions"]}
    colors = {"signal_eol_reproduction": "#78838e", "signal_logrul": "#2474b5",
              "signal_plus_estimated_soh_logrul": "#15916b", "observed_soh_logrul": "#cf8129"}
    names = {"signal_eol_reproduction": "Original: absolute EOL target",
             "signal_logrul": "Same signals: log remaining life",
             "signal_plus_estimated_soh_logrul": "Signals + cross-fitted SOH estimate",
             "observed_soh_logrul": "Measured SOH history (diagnostic only)"}
    fractions = (20, 40, 60, 80)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for mode in colors:
        for axis, metric, label in zip(axes, ("rul_mape_percent", "rul_mae_cycles"),
                                       ("RUL MAPE (%)", "RUL MAE (cycles)")):
            axis.plot(fractions, [metrics[mode][str(f)][metric] for f in fractions],
                      marker="o", lw=2, linestyle="--" if mode == "observed_soh_logrul" else "-",
                      color=colors[mode], label=names[mode])
            axis.set_xticks(fractions)
            axis.set_xlabel("Observed fraction of true event life (%)")
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("MIT: same 43 event cells, same five folds and observation origins", fontsize=13)
    fig.tight_layout(rect=(0, .13, 1, .96))
    fig.savefig(OUT / "rul_target_and_health_comparison.png", dpi=175)
    plt.close(fig)

    comparisons = {}
    for candidate, baseline in (
            ("signal_logrul", "signal_eol_reproduction"),
            ("signal_plus_estimated_soh_logrul", "signal_logrul"),
            ("observed_soh_logrul", "signal_logrul")):
        new, old = ape_index(predictions[candidate]), ape_index(predictions[baseline])
        if new.keys() != old.keys():
            raise ValueError("models do not cover the same cell/origin population")
        comparison = {}
        for fraction in fractions:
            keys = sorted(key for key in new if abs(key[1] - fraction / 100) < 1e-9)
            delta = np.array([new[key] - old[key] for key in keys])
            rng = np.random.default_rng(20928)
            bootstrap = rng.choice(delta, (5000, len(delta)), replace=True).mean(axis=1)
            comparison[str(fraction)] = {
                "mean_rul_mape_delta_pp": float(delta.mean()),
                "within_cohort_cell_bootstrap95_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
                "cells_improved": int(np.sum(delta < 0)), "cells": len(delta),
            }
        comparisons[f"{candidate}_minus_{baseline}"] = comparison
    (OUT / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
    lines = ["# MIT exact-80 RUL: target expression and health information", "",
             "All routes use the same 43 observed-event cells in MIT Batch 2, five whole-cell folds,",
             "and the same retrospective 20/40/60/80% origins. The other 91 cells are right-censored",
             "for this endpoint and do not enter these point-error comparisons. These inspected",
             "development results do not establish cross-batch or independent acceptance.", "",
             "## RUL MAPE by observed-life checkpoint", "",
             "| Route | 20% | 40% | 60% | 80% |",
             "| --- | ---: | ---: | ---: | ---: |"]
    labels = {**names,
              "signal_plus_observed_soh_logrul": "Signals + measured SOH history (diagnostic)",
              "estimated_soh_logrul": "Estimated SOH history only"}
    for mode in metrics:
        lines.append("| " + labels[mode] + " | " + " | ".join(
            f"{metrics[mode][str(f)]['rul_mape_percent']:.2f}%" for f in fractions) + " |")
    lines += ["", "## What the comparison isolates", "",
              "1. The old signal-to-absolute-EOL ExtraTrees is retrained and its predictions",
              "   reproduce the archived original to within 1e-7 cycles. The next arm retains",
              "   the same inputs, rows, cell weights and tree hyperparameters but changes the",
              "   target to log1p(EOL - current cycle); the inverse transform enforces nonnegative RUL.",
              "2. The measured-SOH arms use only current/past full-discharge diagnostics. These",
              "   are richer input scenarios, not 300-second online performance or mathematical upper bounds.",
              "3. The estimated-SOH arms use three inner whole-cell folds inside each outer fold",
              "   to generate RUL training features. Outer test cells train neither the SOH estimator",
              "   nor the RUL regressor. The final outer SOH estimator trains on all outer training cells.",
              "   Inference health features never read measured SOH or capacity-quality flags.", "",
              f"The nested SOH estimator has {nested['soh_pre_eol_cell_macro_mape_percent']:.3f}%",
              "cell-macro current-SOH MAPE on the scored **pre-EOL portion of these 43 event cells**.",
              "This restricted cohort score must not be generalized to all 134 MIT cells or other datasets.",
              "The estimator uses at most 128 training SOH labels per training cell; all available",
              "causal signal histories are used to create predictions. No target SOH label enters inference.", "",
              "## Remaining limitations", "",
              "The added estimated-health features mainly improve the late checkpoint; they do",
              "not consistently improve early-life predictions. These models predict remaining life",
              "directly; joint consistency with a full future SOH trajectory remains to be implemented.",
              "Calibrated individual prediction intervals are not yet provided. The paired bootstrap",
              "in `paired_comparisons.json` describes average error differences within this cohort,",
              "not uncertainty for a new battery or operating condition.", "",
              "## Reproduce", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -m scripts.diagnose_mit_health_rul",
              ".\\.venv\\Scripts\\python.exe -m scripts.diagnose_mit_estimated_health_rul",
              ".\\.venv\\Scripts\\python.exe -m scripts.report_mit_health_rul", "```", "",
              "`diagnostic.json` contains all splits, metrics and point predictions. Each `fold*/`",
              "contains saved SOH/RUL models and the aligned inner/outer SOH predictions.",
              "The measured-input comparison models are in `../mit_health_information_diagnostic/`.", "",
              "![RUL comparison](rul_target_and_health_comparison.png)", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "README.md")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
