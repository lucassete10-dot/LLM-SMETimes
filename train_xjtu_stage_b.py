"""Pre-registered XJTU protocol-held-out Stage-B development comparison.

Batch 2 (3C charge) is never used for fitting or model selection. Batch 3
is the validation condition; the remaining four batches form source training.
This is a single cross-condition development split, not independent acceptance.
"""

import json
from pathlib import Path

import joblib
import torch

from battery_smetimes.data.battery_dataset import (
    CausalWindowDataset, WaveformWindowDataset, fit_source_imputer,
    fit_source_scaler,
)
from battery_smetimes.models.temporal import BatteryHealthGRU, WaveformHealthGRU
from battery_v2_features import load_cell, load_manifest
from train_soh_v2 import fit as fit_tree
from train_xjtu_stage_a import (
    HORIZONS, LOOKBACK, SEED, evaluate_neural, evaluate_tree,
    evaluation_score, train_neural,
)


OUT = Path("battery_models_v3/xjtu_stage_b_protocol_holdout")
WAVEFORMS = Path("data/interim/xjtu_waveforms")
TRAIN_GROUPS = ("batch1", "batch4", "batch5", "batch6")
VALIDATION_GROUP = "batch3"
TEST_GROUP = "batch2"


def protocol_split(records):
    groups = {
        "train": set(TRAIN_GROUPS),
        "validation": {VALIDATION_GROUP},
        "test": {TEST_GROUP},
    }
    if set.union(*groups.values()) != {row["group"] for row in records}:
        raise ValueError("XJTU protocol groups differ from pre-registered split")
    result = {part: sorted(row["cell_id"] for row in records
                           if row["group"] in selected)
              for part, selected in groups.items()}
    if any(not ids for ids in result.values()):
        raise ValueError("empty protocol split")
    if len(set().union(*map(set, result.values()))) != len(records):
        raise ValueError("duplicate or missing cell in protocol split")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = [row for row in load_manifest()["cells"]
               if row["dataset"] == "XJTU"]
    split = protocol_split(records)
    cells = {row["cell_id"]: load_cell(row) for row in records}
    by_id = {row["cell_id"]: row for row in records}
    imputer = fit_source_imputer(cells, split["train"])
    scaler = fit_source_scaler(cells, split["train"], imputer)
    joblib.dump({"imputer": imputer, "scaler": scaler,
                 "train_cell_ids": split["train"]}, OUT / "source_preprocessing.joblib")
    (OUT / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    common = dict(cells=cells, records_by_id=by_id, imputer=imputer,
                  scaler=scaler, lookback=LOOKBACK, future_horizons=HORIZONS)
    summary = {part: CausalWindowDataset(cell_ids=ids, **common)
               for part, ids in split.items()}
    waveform = {part: WaveformWindowDataset(cell_ids=ids,
                                            waveform_dir=WAVEFORMS, **common)
                for part, ids in split.items()}
    for part in split:
        if summary[part].rows != waveform[part].rows:
            raise ValueError(f"summary/waveform mismatch: {part}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "status": "single development protocol holdout; test batch not used for selection",
        "dataset": "XJTU",
        "groups": {"train": TRAIN_GROUPS, "validation": VALIDATION_GROUP,
                   "test": TEST_GROUP},
        "split_cell_counts": {part: len(ids) for part, ids in split.items()},
        "future_horizons_cycles": HORIZONS,
        "seed": SEED,
        "models": {},
    }
    tree = fit_tree(cells, split["train"], "extra_trees_history")
    joblib.dump(tree, OUT / "extra_trees.joblib")
    results["models"]["extra_trees_persistence"] = {
        part: evaluate_tree(tree, summary[part])
        for part in ("validation", "test")}
    print("extra_trees_persistence", {
        part: round(results["models"]["extra_trees_persistence"][part]
                    ["current_cell_macro_mape_percent"], 3)
        for part in ("validation", "test")}, flush=True)
    for name, datasets, factory in (
        ("summary_gru_physics", summary, BatteryHealthGRU),
        ("waveform_gru_physics", waveform, WaveformHealthGRU),
    ):
        model, training = train_neural(datasets["train"],
                                       datasets["validation"],
                                       physics=True, device=device,
                                       model_factory=factory)
        torch.save({"state_dict": model.state_dict(),
                    "signal_width": datasets["train"][0]["signal"].shape[-1],
                    "physics_future": True, "train_cell_ids": split["train"]},
                   OUT / f"{name}.pt")
        results["models"][name] = {
            "training": training,
            "validation": evaluate_neural(model, datasets["validation"], device),
            "test": evaluate_neural(model, datasets["test"], device),
        }
        print(name, {part: round(results["models"][name][part]
                                 ["current_cell_macro_mape_percent"], 3)
                     for part in ("validation", "test")}, flush=True)
    results["validation_selected"] = min(results["models"], key=lambda name:
        evaluation_score(results["models"][name]["validation"]))
    (OUT / "development_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("Selected on validation:", results["validation_selected"], flush=True)


if __name__ == "__main__":
    main()
