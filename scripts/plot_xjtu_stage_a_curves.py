"""Visualize identical XJTU development test cells across model families."""

import json
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from battery_smetimes.data.battery_dataset import (CausalWindowDataset,
                                                   WaveformWindowDataset)
from battery_smetimes.models.temporal import BatteryHealthGRU, WaveformHealthGRU
from battery_v2_features import load_cell, load_manifest
from train_soh_v2 import predict as predict_tree


BASE = Path("battery_models_v3/xjtu_stage_a")
WAVE = Path("battery_models_v3/xjtu_waveform_stage_a")


def neural_predictions(dataset, path, model_class, device):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = model_class(bundle["signal_width"],
                        physics_future=bundle["physics_future"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    estimates = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=256, shuffle=False):
            extra = ({"waveform": batch["waveform"].to(device)}
                     if "waveform" in batch else {})
            result = model(batch["signal"].to(device), batch["mask"].to(device),
                           batch["metadata"].to(device), **extra)
            estimates.extend(result["current_soh"].cpu().numpy())
    return np.asarray(estimates)


def main():
    split = json.loads((BASE / "split.json").read_text(encoding="utf-8"))
    prep = joblib.load(BASE / "source_preprocessing.joblib")
    records = [row for row in load_manifest()["cells"]
               if row["dataset"] == "XJTU" and row["cell_id"] in split["test"]]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    by_id = {row["cell_id"]: row for row in records}
    common = dict(cells=cells, records_by_id=by_id, cell_ids=split["test"],
                  imputer=prep["imputer"], scaler=prep["scaler"], lookback=16)
    summary = CausalWindowDataset(**common)
    waveform = WaveformWindowDataset(**common,
                                     waveform_dir="data/interim/xjtu_waveforms")
    if summary.rows != waveform.rows:
        raise ValueError("summary/waveform test row mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_pred = neural_predictions(summary, BASE / "gru_physics.pt",
                                      BatteryHealthGRU, device)
    waveform_pred = neural_predictions(waveform, WAVE / "waveform_gru_physics.pt",
                                       WaveformHealthGRU, device)
    tree = joblib.load(BASE / "extra_trees.joblib")
    tree_pred = {key: predict_tree(tree, cell) for key, cell in cells.items()}
    by_cell = defaultdict(list)
    for index, (key, position) in enumerate(summary.rows):
        by_cell[key].append((int(cells[key].cycle[position]),
                             float(cells[key].soh[position]),
                             float(tree_pred[key][position]),
                             float(summary_pred[index]),
                             float(waveform_pred[index])))
    fig, axes = plt.subplots(4, 2, figsize=(16, 15))
    colors = {"Measured": "#101010", "ExtraTrees": "#e58620",
              "Summary GRU + physics": "#3269b3",
              "Waveform GRU + physics": "#159779"}
    for axis, key in zip(axes.flat, sorted(by_cell)):
        array = np.asarray(sorted(by_cell[key]))
        for column, name in enumerate(colors, start=1):
            axis.plot(array[:, 0], array[:, column], color=colors[name],
                      linewidth=2 if name == "Measured" else 1.5,
                      alpha=.9, label=name)
        mape = np.mean(100 * np.abs(array[:, 4] - array[:, 1]) / array[:, 1])
        axis.set_title(f"{key} | waveform MAPE {mape:.2f}%", fontsize=10)
        axis.set_ylim(.78, 1.08)
        axis.grid(alpha=.2)
        axis.axhline(.8, color="#b6272d", linestyle="--", linewidth=.8)
        axis.set_xlabel("Source cycle")
        axis.set_ylabel("SOH")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .973),
               ncol=4, frameon=False)
    fig.suptitle("XJTU held-out development cells: current SOH estimates (not blind final test)",
                 y=.997, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    target = WAVE / "test_cell_curves.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
