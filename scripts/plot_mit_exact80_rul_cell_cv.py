"""Plot whole-cell MIT exact-80% RUL development out-of-fold diagnostics."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "battery_models_v3/mit_exact80_rul_stage_a/five_fold_cell_oof.json"


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    survival_source = SOURCE.with_name("censored_survival_five_fold_cv.json")
    survival = (json.loads(survival_source.read_text(encoding="utf-8"))
                if survival_source.exists() else None)
    names = ("median_train_eol", "ridge", "extra_trees")
    labels = ("Train-fold median life", "Ridge", "ExtraTrees")
    colors = ("#858585", "#366db0", "#d97d18")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fractions = (20, 40, 60, 80)
    for name, label, color in zip(names, labels, colors):
        values = [report["aggregate"][name][str(fraction)]["rul_mape_percent"]
                  for fraction in fractions]
        axes[0].plot(fractions, values, marker="o", color=color, label=label)
    if survival:
        for name, label, color in (
                ("aft_events_only", "AFT, events only", "#159779"),
                ("aft_with_censored", "AFT + other-batch censored", "#bd4c60")):
            values = [survival["aggregate"][name]["point_metrics"][str(fraction)]
                      ["rul_mape_percent"] for fraction in fractions]
            axes[0].plot(fractions, values, marker="s", color=color, label=label)
    axes[0].set_xticks(fractions)
    axes[0].set_xlabel("Observed fraction of true lifetime (retrospective checkpoint)")
    axes[0].set_ylabel("Remaining-life MAPE (%)")
    axes[0].set_title("Five-fold held-out-cell RUL error")
    axes[0].grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)

    rows = [row for row in report["oof_predictions"]["extra_trees"]
            if row["fraction"] == .8]
    actual = np.asarray([row["actual_eol_cycle"] for row in rows])
    predicted = np.asarray([row["predicted_eol_cycle"] for row in rows])
    axes[1].scatter(actual, predicted, c="#d97d18", alpha=.8)
    axes[1].plot([140, 760], [140, 760], "--", color="#444444", linewidth=1)
    extreme = sorted(range(len(rows)), key=lambda index: abs(predicted[index] - actual[index]),
                     reverse=True)[:3]
    for index in extreme:
        axes[1].annotate(rows[index]["cell_id"], (actual[index], predicted[index]),
                         xytext=(4, 5), textcoords="offset points", fontsize=8)
    axes[1].set_xlim(140, 760)
    axes[1].set_ylim(140, 760)
    axes[1].set_xlabel("True 80%-SOH EOL cycle")
    axes[1].set_ylabel("ExtraTrees predicted EOL cycle")
    axes[1].set_title("At 80% lifetime observed: early failures are missed")
    axes[1].grid(alpha=.25)
    fig.suptitle("MIT Batch 2 exact-80% RUL: 43 observed events, development CV only",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    target = SOURCE.with_name("five_fold_rul_diagnostics.png")
    fig.savefig(target, dpi=170)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
