"""Charge-only XJTU protocol-held-out development check."""

import json
from pathlib import Path

import joblib
import torch

from battery_smetimes.data.battery_dataset import (
    CausalWindowDataset, WaveformWindowDataset, fit_source_imputer,
    fit_source_scaler,
)
from battery_smetimes.models.temporal import BatteryHealthGRU
from battery_v2_features import load_cell, load_manifest
from train_xjtu_charge_stage_a import (
    VARIANT, WAVEFORMS, charge_only_records, charge_waveform_factory,
    fit_charge_tree,
)
from train_xjtu_stage_a import (
    HORIZONS, LOOKBACK, SEED, evaluate_neural, evaluate_tree,
    evaluation_score, train_neural,
)


BASE = Path("battery_models_v3/xjtu_stage_b_protocol_holdout")
OUT = Path("battery_models_v3/xjtu_charge_stage_b")


def main():
    split = json.loads((BASE / "split.json").read_text(encoding="utf-8"))
    records = [row for row in load_manifest()["cells"] if row["dataset"] == "XJTU"]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    metadata = charge_only_records(records)
    imputer = fit_source_imputer(cells, split["train"], variant=VARIANT)
    scaler = fit_source_scaler(cells, split["train"], imputer, variant=VARIANT)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"imputer": imputer, "scaler": scaler,
                 "train_cell_ids": split["train"], "variant": VARIANT,
                 "discharge_rate_masked": True},
                OUT / "charge_source_preprocessing.joblib")
    common = dict(cells=cells, records_by_id=metadata, imputer=imputer,
                  scaler=scaler, lookback=LOOKBACK, variant=VARIANT,
                  future_horizons=HORIZONS)
    summary = {part: CausalWindowDataset(cell_ids=ids, **common)
               for part, ids in split.items()}
    waveform = {part: WaveformWindowDataset(cell_ids=ids,
                                            waveform_dir=WAVEFORMS,
                                            waveform_stage="charge", **common)
                for part, ids in split.items()}
    if any(summary[part].rows != waveform[part].rows for part in split):
        raise ValueError("charge-only protocol split row mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "status": "XJTU charge-only cross-condition development; Batch 2 already inspected",
        "split_source": str(BASE / "split.json"),
        "train_group": "batches 1,4,5,6", "validation_group": "batch3",
        "test_group": "batch2", "seed": SEED,
        "feature_variant": VARIANT,
        "selection_metric": "validation current MAPE + 0.2 mean future MAPE",
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
    for name, datasets, factory in (
        ("charge_summary_gru_physics", summary, BatteryHealthGRU),
        ("charge_waveform_gru_physics", waveform, charge_waveform_factory),
    ):
        model, training = train_neural(datasets["train"],
                                       datasets["validation"],
                                       physics=True, device=device,
                                       model_factory=factory)
        torch.save({"state_dict": model.state_dict(),
                    "signal_width": datasets["train"][0]["signal"].shape[-1],
                    "physics_future": True,
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
    result["validation_selected_charge_only"] = min(result["models"], key=lambda name:
        evaluation_score(result["models"][name]["validation"]))
    previous = json.loads((BASE / "development_results.json").read_text(encoding="utf-8"))
    result["full_charge_plus_discharge_reference"] = {
        part: {"current_cell_macro_mape_percent":
               previous["models"]["waveform_gru_physics"][part]
               ["current_cell_macro_mape_percent"],
               "future_cell_macro_mape_percent":
               previous["models"]["waveform_gru_physics"][part]
               ["future_cell_macro_mape_percent"]}
        for part in ("validation", "test")}
    (OUT / "development_results.json").write_text(json.dumps(result, indent=2),
                                                     encoding="utf-8")
    print("Selected charge-only on validation:",
          result["validation_selected_charge_only"], flush=True)


if __name__ == "__main__":
    main()
