"""Same XJTU whole-cell split, now with a depthwise V/I/T/Q Cycle Encoder."""

import json
from pathlib import Path

import joblib
import torch

from battery_smetimes.data.battery_dataset import WaveformWindowDataset
from battery_smetimes.models.temporal import WaveformHealthGRU
from battery_v2_features import load_cell, load_manifest
from train_xjtu_stage_a import (HORIZONS, LOOKBACK, SEED, evaluate_neural,
                                evaluation_score, train_neural)


BASE = Path("battery_models_v3/xjtu_stage_a")
OUT = Path("battery_models_v3/xjtu_waveform_stage_a")
WAVEFORMS = Path("data/interim/xjtu_waveforms")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((BASE / "split.json").read_text(encoding="utf-8"))
    preprocessing = joblib.load(BASE / "source_preprocessing.joblib")
    if preprocessing["train_cell_ids"] != split["train"]:
        raise ValueError("source preprocessing was not fit on this training split")
    source = json.loads((WAVEFORMS / "manifest.json").read_text(encoding="utf-8"))
    if len(source["cells"]) != 55 or source["label_source_loaded"]:
        raise ValueError("XJTU waveforms have not passed label-free 55-cell QA")
    manifest = load_manifest()
    records = [row for row in manifest["cells"] if row["dataset"] == "XJTU"]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    records_by_id = {row["cell_id"]: row for row in records}
    datasets = {part: WaveformWindowDataset(
        cells, records_by_id, split[part], preprocessing["imputer"],
        scaler=preprocessing["scaler"], waveform_dir=WAVEFORMS,
        lookback=LOOKBACK, future_horizons=HORIZONS) for part in split}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = json.loads((BASE / "development_results.json").read_text(encoding="utf-8"))
    result = {
        "status": "development ablation, same already-inspected XJTU cells as summary model",
        "dataset": "XJTU", "raw_waveform": "first 120 seconds charge/discharge V/I/T/derived Q",
        "split_source": str(BASE / "split.json"),
        "preprocessing_source": str(BASE / "source_preprocessing.joblib"),
        "source_mat_variable_loaded": "data only, never summary",
        "future_horizons_cycles": HORIZONS, "device": str(device), "seed": SEED,
        "comparison_reference": {name: {
            part: {"current_cell_macro_mape_percent":
                   row[part]["current_cell_macro_mape_percent"],
                   "future_cell_macro_mape_percent":
                   row[part]["future_cell_macro_mape_percent"]}
            for part in ("validation", "test")}
            for name, row in baseline["models"].items()},
        "models": {},
    }
    for name, physics in (("waveform_gru_direct", False),
                          ("waveform_gru_physics", True)):
        model, training = train_neural(datasets["train"], datasets["validation"],
                                       physics=physics, device=device,
                                       model_factory=WaveformHealthGRU)
        torch.save({"state_dict": model.state_dict(), "signal_width":
                    datasets["train"][0]["signal"].shape[-1],
                    "physics_future": physics, "train_cell_ids": split["train"]},
                   OUT / f"{name}.pt")
        result["models"][name] = {
            "training": training,
            "validation": evaluate_neural(model, datasets["validation"], device),
            "test": evaluate_neural(model, datasets["test"], device),
        }
        print(name, {part: round(result["models"][name][part]
                                 ["current_cell_macro_mape_percent"], 3)
                     for part in ("validation", "test")}, flush=True)
    result["validation_selected"] = min(result["models"], key=lambda name:
        evaluation_score(result["models"][name]["validation"]))
    result["beats_summary_validation"] = bool(
        evaluation_score(result["models"][result["validation_selected"]]["validation"])
        < min(evaluation_score(row["validation"]) for row in baseline["models"].values()))
    (OUT / "development_results.json").write_text(json.dumps(result, indent=2),
                                                     encoding="utf-8")
    print("Selected on validation:", result["validation_selected"],
          "beats summary:", result["beats_summary_validation"], flush=True)


if __name__ == "__main__":
    main()
