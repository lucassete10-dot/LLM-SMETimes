"""XJTU unseen-cell development baseline: ExtraTrees vs GRU vs GRU+physics.

Only signal history available at the observation cycle and fixed protocol
metadata enter inference. Future capacity appears solely as a training label.
The split is whole-cell 70/15/15, group-stratified, and saved before scoring.
This is *within-condition* development, not a cross-strategy or blind test.
"""

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from battery_smetimes.data.battery_dataset import (
    CausalWindowDataset, DomainCellBalancedSampler, fit_source_imputer,
    fit_source_scaler,
)
from battery_smetimes.data.split import cellwise_split
from battery_smetimes.models.temporal import BatteryHealthGRU
from battery_v2_features import load_cell, load_manifest
from train_soh_v2 import fit as fit_tree, predict as predict_tree


OUT = Path("battery_models_v3/xjtu_stage_a")
HORIZONS = (10, 50, 100)
LOOKBACK = 16
SEED = 2026
MAX_EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 128


def metric_rows(rows):
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[row["cell_id"]].append(row)
    cells = []
    for cell_id, observations in sorted(by_cell.items()):
        target = np.asarray([row["target"] for row in observations])
        estimate = np.asarray([row["estimate"] for row in observations])
        if not np.all(np.isfinite(estimate)):
            raise ValueError(f"nonfinite current SOH prediction: {cell_id}")
        future = {}
        for slot, horizon in enumerate(HORIZONS):
            truth = np.asarray([row["future_target"][slot] for row in observations])
            prediction = np.asarray([row["future_estimate"][slot] for row in observations])
            keep = np.isfinite(truth)
            future[str(horizon)] = {
                "observations": int(keep.sum()),
                "mape_percent": (float(np.mean(100 * np.abs(prediction[keep] - truth[keep]) /
                                               truth[keep])) if keep.any() else None),
            }
        cells.append({
            "cell_id": cell_id,
            "observations": len(observations),
            "current_mape_percent": float(np.mean(100 * np.abs(estimate - target) / target)),
            "current_mae_soh": float(np.mean(np.abs(estimate - target))),
            "future": future,
        })
    current = [row["current_mape_percent"] for row in cells]
    return {
        "cells": len(cells), "observations": len(rows),
        "current_cell_macro_mape_percent": float(np.mean(current)),
        "current_p90_cell_mape_percent": float(np.quantile(current, .9)),
        "current_worst_cell_mape_percent": float(np.max(current)),
        "future_cell_macro_mape_percent": {
            str(h): float(np.mean([row["future"][str(h)]["mape_percent"] for row in cells
                                   if row["future"][str(h)]["mape_percent"] is not None]))
            for h in HORIZONS},
        "by_cell": cells,
    }


def evaluation_score(metrics):
    """Validation-only model selection balances current and future errors."""
    future = np.mean(list(metrics["future_cell_macro_mape_percent"].values()))
    return metrics["current_cell_macro_mape_percent"] + .2 * future


def evaluate_neural(model, dataset, device):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            extra = ({"waveform": batch["waveform"].to(device)}
                     if "waveform" in batch else {})
            if "cycle_context" in batch:
                extra["cycle_context"] = batch["cycle_context"].to(device)
            result = model(batch["signal"].to(device), batch["mask"].to(device),
                           batch["metadata"].to(device), HORIZONS, **extra)
            current = result["current_soh"].cpu().numpy()
            future = result.get("future_soh")
            if future is None:
                future = np.repeat(current[:, None], len(HORIZONS), axis=1)
            else:
                future = future.cpu().numpy()
            for index, cell_id in enumerate(batch["cell_id"]):
                rows.append({
                    "cell_id": cell_id,
                    "target": float(batch["target_soh"][index]),
                    "estimate": float(current[index]),
                    "future_target": batch["future_soh"][index].numpy().tolist(),
                    "future_estimate": future[index].tolist(),
                })
    return metric_rows(rows)


def evaluate_tree(bundle, dataset):
    predictions = {key: predict_tree(bundle, cell)
                   for key, cell in dataset.cells.items()}
    rows = []
    for index, (cell_id, position) in enumerate(dataset.rows):
        sample = dataset[index]
        value = float(predictions[cell_id][position])
        rows.append({
            "cell_id": cell_id,
            "target": float(sample["target_soh"]),
            "estimate": value,
            "future_target": sample["future_soh"].numpy().tolist(),
            "future_estimate": [value] * len(HORIZONS),
        })
    return metric_rows(rows)


def train_neural(train, validation, *, physics, device, model_factory=BatteryHealthGRU,
                 seed=SEED, epoch_callback=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = model_factory(train[0]["signal"].shape[-1],
                          physics_future=physics).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sampler = DomainCellBalancedSampler(train, len(train), seed=seed)
    loader = DataLoader(train, batch_size=BATCH_SIZE, sampler=sampler)
    best_score, best_state, best_epoch = float("inf"), None, None
    history = []
    for epoch in range(MAX_EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        losses = []
        for batch in loader:
            signal = batch["signal"].to(device)
            mask = batch["mask"].to(device)
            metadata = batch["metadata"].to(device)
            target = batch["target_soh"].to(device)
            weight = batch["sample_weight"].to(device)
            extra = ({"waveform": batch["waveform"].to(device)}
                     if "waveform" in batch else {})
            if "cycle_context" in batch:
                extra["cycle_context"] = batch["cycle_context"].to(device)
            output = model(signal, mask, metadata, HORIZONS, **extra)
            current_loss = F.huber_loss(output["current_soh"], target,
                                        delta=.03, reduction="none")
            loss = (current_loss * weight).sum() / weight.sum()
            if physics:
                future_target = batch["future_soh"].to(device)
                available = batch["future_available"].to(device)
                safe_target = torch.where(available, future_target,
                                          output["future_soh"].detach())
                future_loss = F.huber_loss(output["future_soh"], safe_target,
                                           delta=.03, reduction="none")
                future_weight = available.float() * weight[:, None]
                loss = loss + .25 * (future_loss * future_weight).sum() / \
                    future_weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate_neural(model, validation, device)
        criterion = evaluation_score(metrics)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_current_mape_percent":
                        metrics["current_cell_macro_mape_percent"],
                        "validation_future_mape_percent":
                        metrics["future_cell_macro_mape_percent"],
                        "validation_selection_score": criterion})
        if epoch_callback is not None:
            epoch_callback(history[-1])
        if criterion < best_score - 1e-5:
            best_score = criterion
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        if epoch + 1 - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "history": history,
                   "parameters": sum(p.numel() for p in model.parameters())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    records = [row for row in manifest["cells"] if row["dataset"] == "XJTU"]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    records_by_id = {row["cell_id"]: row for row in records}
    split = cellwise_split(records, "XJTU", seed=SEED)
    (OUT / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    imputer = fit_source_imputer(cells, split["train"])
    scaler = fit_source_scaler(cells, split["train"], imputer)
    joblib.dump({"imputer": imputer, "scaler": scaler,
                 "train_cell_ids": split["train"]}, OUT / "source_preprocessing.joblib")
    datasets = {part: CausalWindowDataset(
        cells, records_by_id, split[part], imputer, scaler=scaler,
        lookback=LOOKBACK, future_horizons=HORIZONS) for part in split}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "status": "development; held-out cells not an external blind test",
        "dataset": "XJTU", "split_type": "within-condition whole-cell 70/15/15",
        "current_target": "measured comparable-discharge capacity SOH",
        "future_horizons_cycles": HORIZONS,
        "rul_status": "not evaluated; no confirmed per-cell 80%-SOH event in XJTU v2",
        "seed": SEED, "lookback": LOOKBACK, "device": str(device),
        "split_cell_counts": {part: len(ids) for part, ids in split.items()},
        "models": {},
    }
    tree = fit_tree(cells, split["train"], "extra_trees_history")
    joblib.dump(tree, OUT / "extra_trees.joblib")
    result["models"]["extra_trees_persistence"] = {
        part: evaluate_tree(tree, datasets[part]) for part in ("validation", "test")}
    print("ExtraTrees", {part: round(result["models"]["extra_trees_persistence"][part]
                                    ["current_cell_macro_mape_percent"], 3)
                         for part in ("validation", "test")}, flush=True)
    for name, physics in (("gru_direct", False), ("gru_physics", True)):
        model, training = train_neural(datasets["train"], datasets["validation"],
                                       physics=physics, device=device)
        torch.save({"state_dict": model.state_dict(), "signal_width":
                    datasets["train"][0]["signal"].shape[-1],
                    "physics_future": physics, "train_cell_ids": split["train"]},
                   OUT / f"{name}.pt")
        result["models"][name] = {"training": training,
                                   "validation": evaluate_neural(model, datasets["validation"], device),
                                   "test": evaluate_neural(model, datasets["test"], device)}
        print(name, {part: round(result["models"][name][part]
                                 ["current_cell_macro_mape_percent"], 3)
                     for part in ("validation", "test")}, flush=True)
    result["validation_selected"] = min(result["models"], key=lambda name:
        evaluation_score(result["models"][name]["validation"]))
    (OUT / "development_results.json").write_text(json.dumps(result, indent=2),
                                                     encoding="utf-8")
    print("Selected on validation:", result["validation_selected"], flush=True)


if __name__ == "__main__":
    main()
