"""Render current held-out battery evaluation results without retraining."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "battery_models_v2"
OUTPUT = RESULTS / "evaluation_plots"
OUTPUT.mkdir(exist_ok=True)
DOMAINS = ("NASA", "CALCE", "MIT", "XJTU")


def rul_at_cycle(checkpoint=50):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    for ax, domain in zip(axes.flat, DOMAINS):
        data = np.load(RESULTS / "rul" / domain.lower() / "development_oof_predictions.npz")
        cycles, endpoints = [], []
        for cell in np.unique(data["cell_id"]):
            positions = np.flatnonzero(
                (data["cell_id"] == cell) & (data["cycle"] == checkpoint)
            )
            if len(positions) != 1:
                continue
            idx = positions[0]
            if data["endpoint"][idx] <= checkpoint:
                continue
            cycles.append(float(data["cycle"][idx] + data["estimate"][idx]))
            endpoints.append(float(data["endpoint"][idx]))
        actual = np.asarray(endpoints)
        predicted = np.asarray(cycles)
        mape = np.mean(np.abs(predicted - actual) / actual) * 100 if len(actual) else np.nan
        if len(actual):
            lower = min(np.min(actual), np.min(predicted))
            upper = max(np.max(actual), np.max(predicted))
            pad = (upper - lower) * 0.07 or 1
            ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad],
                    color="#6b7280", linestyle="--", linewidth=1.4, label="Perfect prediction")
            ax.scatter(actual, predicted, color="#1976a3", alpha=0.7, s=35)
            ax.set_xlim(lower - pad, upper + pad)
            ax.set_ylim(lower - pad, upper + pad)
        ax.set_title(f"{domain}  |  n={len(actual)}  |  EOL MAPE={mape:.1f}%")
        ax.set_xlabel("Observed 80%-SOH EOL cycle")
        ax.set_ylabel("Predicted 80%-SOH EOL cycle")
        ax.grid(alpha=0.2)
    fig.suptitle(f"RUL evaluation: predicted vs observed 80%-SOH endpoint at cycle {checkpoint}", fontsize=14)
    fig.savefig(OUTPUT / f"rul_eol_at_cycle_{checkpoint}.png", dpi=180)
    plt.close(fig)


def forecast_horizons():
    data = json.loads((RESULTS / "trajectory" / "future_soh_mape.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x = np.arange(len(DOMAINS))
    for offset, horizon, color in ((-0.19, "50", "#1976a3"), (0.19, "100", "#e28a26")):
        values = [data["domains"][domain]["summary"][horizon]["equal_cell_mape_percent"] for domain in DOMAINS]
        counts = [data["domains"][domain]["summary"][horizon]["scored_cells"] for domain in DOMAINS]
        bars = ax.bar(x + offset, values, width=0.36, label=f"{horizon}-cycle ahead", color=color)
        for bar, value, count in zip(bars, values, counts):
            ax.annotate(f"{value:.2f}%\nn={count}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.axhline(6, linestyle="--", color="#bb2a39", label="6% target")
    ax.set_xticks(x, DOMAINS)
    ax.set_ylim(0, max(11, ax.get_ylim()[1] * 1.13))
    ax.set_ylabel("Future SOH MAPE (%)")
    ax.set_title("Held-out rolling-origin future SOH forecasts")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUTPUT / "future_soh_50_100_cycle_mape.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    rul_at_cycle()
    forecast_horizons()
    print(OUTPUT)
