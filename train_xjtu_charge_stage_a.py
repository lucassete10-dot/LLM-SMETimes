"""Charge-prefix-only XJTU SOH ablation on the frozen whole-cell split."""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from battery_smetimes.data.battery_dataset import (
    CausalWindowDataset, WaveformWindowDataset, fit_source_imputer,
    fit_source_scaler,
)
from battery_smetimes.models.temporal import BatteryHealthGRU, WaveformHealthGRU
from battery_v2_features import load_cell, load_manifest
from train_xjtu_stage_a import (
    HORIZONS, LOOKBACK, SEED, evaluate_neural, evaluate_tree,
    evaluation_score, train_neural,
)


BASE = Path("battery_models_v3/xjtu_stage_a")
FULL_WAVE = Path("battery_models_v3/xjtu_waveform_stage_a")
OUT = Path("battery_models_v3/xjtu_charge_stage_a")
WAVEFORMS = Path("data/interim/xjtu_waveforms")
VARIANT = "charge_signal_history"


def charge_only_records(records):
    """Retain known charge/ambient protocol, mask discharge-side metadata."""
    return {row["cell_id"]: {**row, "discharge_rate_c": None}
            for row in records}


def fit_charge_tree(cells, train_ids, imputer):
    features, targets, weights = [], [], []
    for key in train_ids:
        cell = cells[key]
        keep = cell.soh_available
        if not keep.any():
            raise ValueError(f"no XJTU training SOH labels: {key}")
        features.append(cell.matrix(VARIANT)[keep])
        targets.append(cell.soh[keep])
        quality = np.where(cell.capacity_review_flag[keep], .25, 1.)
        weights.append(quality / quality.sum() / len(train_ids))
    x = imputer.transform(np.concatenate(features))
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=SEED, n_jobs=-1)
    model.fit(x, np.concatenate(targets), sample_weight=np.concatenate(weights))
    return {"name": "extra_trees_charge_only", "variant": VARIANT,
            "imputer": imputer, "scaler": None, "model": model,
            "training_cell_ids": list(train_ids),
            "input_stage": "charge only; no discharge waveform or signal features"}


def charge_waveform_factory(signal_width, *, physics_future):
    return WaveformHealthGRU(signal_width, physics_future=physics_future,
                             stages=1)


def main():
    audit = json.loads((OUT / "feature_order_audit.json").read_text(encoding="utf-8"))
    if audit["cells"] != 55 or audit["diagnostic_cycles"] != 10389:
        raise ValueError("charge-only XJTU feature order has not passed QA")
    split = json.loads((BASE / "split.json").read_text(encoding="utf-8"))
    records = [row for row in load_manifest()["cells"] if row["dataset"] == "XJTU"]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    metadata = charge_only_records(records)
    imputer = fit_source_imputer(cells, split["train"], variant=VARIANT)
    scaler = fit_source_scaler(cells, split["train"], imputer, variant=VARIANT)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"imputer": imputer, "scaler": scaler,
                 "variant": VARIANT, "train_cell_ids": split["train"],
                 "discharge_rate_masked": True},
                OUT / "charge_source_preprocessing.joblib")
    common = dict(cells=cells, records_by_id=metadata, imputer=imputer,
                  scaler=scaler, lookback=LOOKBACK, variant=VARIANT,
                  future_horizons=HORIZONS)
    summary = {part: CausalWindowDataset(cell_ids=ids, **common)
               for part, ids in split.items()}
    waveform = {part: WaveformWindowDataset(
        cell_ids=ids, waveform_dir=WAVEFORMS, waveform_stage="charge", **common)
        for part, ids in split.items()}
    if any(summary[part].rows != waveform[part].rows for part in split):
        raise ValueError("charge summary/waveform alignment mismatch")
    if summary["train"][0]["signal"].shape[-1] != 69 or \
       waveform["train"][0]["waveform"].shape[-3:] != (1, 4, 121):
        raise ValueError("charge-only training contract failed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "status": "XJTU charge-only development ablation, same already-inspected whole-cell split",
        "split_source": str(BASE / "split.json"),
        "feature_variant": VARIANT,
        "waveform": "first 120 seconds of charge V/I/T/integrated Q only",
        "metadata": "ambient and charge rate when known; discharge rate masked",
        "selection_metric": "validation current MAPE + 0.2 mean future MAPE",
        "seed": SEED, "lookback": LOOKBACK, "future_horizons_cycles": HORIZONS,
        "models": {},
    }
    tree = fit_charge_tree(cells, split["train"], imputer)
    joblib.dump(tree, OUT / "charge_extra_trees.joblib")
    result["models"]["charge_extra_trees_persistence"] = {
        part: evaluate_tree(tree, summary[part])
        for part in ("validation", "test")}
    print("charge_extra_trees_persistence", {
        part: round(result["models"]["charge_extra_trees_persistence"][part]
                    ["current_cell_macro_mape_percent"], 3)
        for part in ("validation", "test")}, flush=True)
    for name, datasets, factory, physics in (
        ("charge_summary_gru_direct", summary, BatteryHealthGRU, False),
        ("charge_summary_gru_physics", summary, BatteryHealthGRU, True),
        ("charge_waveform_gru_direct", waveform, charge_waveform_factory, False),
        ("charge_waveform_gru_physics", waveform, charge_waveform_factory, True),
    ):
        model, training = train_neural(datasets["train"],
                                       datasets["validation"],
                                       physics=physics, device=device,
                                       model_factory=factory)
        torch.save({"state_dict": model.state_dict(),
                    "signal_width": datasets["train"][0]["signal"].shape[-1],
                    "physics_future": physics,
                    "waveform_stages": 1 if "waveform" in name else 0,
                    "train_cell_ids": split["train"]},
                   OUT / f"{name}.pt")
        result["models"][name] = {
            "training": training,
            "validation": evaluate_neural(model, datasets["validation"], device),
            "test": evaluate_neural(model, datasets["test"], device),
        }
        print(name, {part: round(result["models"][name][part]
                                 ["current_cell_macro_mape_percent"], 3)
                     for part in ("validation", "test")}, flush=True)
    result["validation_selected_charge_only"] = min(result["models"],
        key=lambda name: evaluation_score(result["models"][name]["validation"]))
    full = json.loads((FULL_WAVE / "development_results.json").read_text(encoding="utf-8"))
    result["full_charge_plus_discharge_reference"] = {
        part: {"current_cell_macro_mape_percent": full["models"]["waveform_gru_physics"]
               [part]["current_cell_macro_mape_percent"],
               "future_cell_macro_mape_percent": full["models"]["waveform_gru_physics"]
               [part]["future_cell_macro_mape_percent"]}
        for part in ("validation", "test")}
    (OUT / "development_results.json").write_text(json.dumps(result, indent=2),
                                                     encoding="utf-8")
    print("Selected charge-only on validation:",
          result["validation_selected_charge_only"], flush=True)


if __name__ == "__main__":
    main()
