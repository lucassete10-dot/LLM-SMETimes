"""Causal TCN checks on the frozen XJTU protocol-held-out split."""

import json
from pathlib import Path

import joblib
import torch

from battery_smetimes.data.battery_dataset import (
    CausalWindowDataset, WaveformWindowDataset,
)
from battery_smetimes.models.temporal import BatteryHealthTCN, WaveformHealthTCN
from battery_v2_features import load_cell, load_manifest
from train_xjtu_stage_a import (
    HORIZONS, LOOKBACK, SEED, evaluate_neural, evaluation_score, train_neural,
)


BASE = Path("battery_models_v3/xjtu_stage_b_protocol_holdout")
OUT = Path("battery_models_v3/xjtu_stage_b_tcn")
WAVEFORMS = Path("data/interim/xjtu_waveforms")


def main():
    split = json.loads((BASE / "split.json").read_text(encoding="utf-8"))
    prep = joblib.load(BASE / "source_preprocessing.joblib")
    if prep["train_cell_ids"] != split["train"]:
        raise ValueError("frozen protocol preprocessing train cells mismatch")
    records = [row for row in load_manifest()["cells"] if row["dataset"] == "XJTU"]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    by_id = {row["cell_id"]: row for row in records}
    common = dict(cells=cells, records_by_id=by_id,
                  imputer=prep["imputer"], scaler=prep["scaler"],
                  lookback=LOOKBACK, future_horizons=HORIZONS)
    summary = {part: CausalWindowDataset(cell_ids=ids, **common)
               for part, ids in split.items()}
    waveform = {part: WaveformWindowDataset(cell_ids=ids,
                                            waveform_dir=WAVEFORMS, **common)
                for part, ids in split.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "status": "XJTU protocol-held-out TCN development comparison; Batch 2 already inspected",
        "split_source": str(BASE / "split.json"),
        "preprocessing_source": str(BASE / "source_preprocessing.joblib"),
        "train_group": "batches 1,4,5,6", "validation_group": "batch3",
        "test_group": "batch2", "seed": SEED,
        "selection_metric": "validation current MAPE + 0.2 mean future MAPE",
        "models": {},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, datasets, factory in (
        ("summary_tcn_physics", summary, BatteryHealthTCN),
        ("waveform_tcn_physics", waveform, WaveformHealthTCN),
    ):
        model, training = train_neural(datasets["train"],
                                       datasets["validation"],
                                       physics=True, device=device,
                                       model_factory=factory)
        torch.save({"state_dict": model.state_dict(),
                    "signal_width": datasets["train"][0]["signal"].shape[-1],
                    "physics_future": True, "train_cell_ids": split["train"]},
                   OUT / f"{name}.pt")
        result["models"][name] = {
            "training": training,
            "validation": evaluate_neural(model, datasets["validation"], device),
            "test": evaluate_neural(model, datasets["test"], device),
        }
        print(name, {part: round(result["models"][name][part]
                                 ["current_cell_macro_mape_percent"], 3)
                     for part in ("validation", "test")}, flush=True)
    prior = json.loads((BASE / "development_results.json").read_text(encoding="utf-8"))
    result["prior_comparison"] = {name: {
        part: {"current_cell_macro_mape_percent":
               row[part]["current_cell_macro_mape_percent"],
               "future_cell_macro_mape_percent":
               row[part]["future_cell_macro_mape_percent"]}
        for part in ("validation", "test")}
        for name, row in prior["models"].items()}
    candidates = {**result["prior_comparison"], **result["models"]}
    result["validation_selected"] = min(candidates, key=lambda name:
        evaluation_score(candidates[name]["validation"]))
    (OUT / "development_results.json").write_text(json.dumps(result, indent=2),
                                                     encoding="utf-8")
    print("Selected on validation:", result["validation_selected"], flush=True)


if __name__ == "__main__":
    main()
