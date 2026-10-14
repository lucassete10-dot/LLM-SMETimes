"""Capacity-free TCN baselines on the frozen XJTU Stage-A whole-cell split."""

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


BASE = Path("battery_models_v3/xjtu_stage_a")
WAVE = Path("battery_models_v3/xjtu_waveform_stage_a")
OUT = Path("battery_models_v3/xjtu_tcn_stage_a")
WAVEFORMS = Path("data/interim/xjtu_waveforms")


def main():
    split = json.loads((BASE / "split.json").read_text(encoding="utf-8"))
    preprocessing = joblib.load(BASE / "source_preprocessing.joblib")
    if preprocessing["train_cell_ids"] != split["train"]:
        raise ValueError("preprocessor was not fitted to frozen training cells")
    records = [row for row in load_manifest()["cells"] if row["dataset"] == "XJTU"]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    by_id = {row["cell_id"]: row for row in records}
    common = dict(cells=cells, records_by_id=by_id,
                  imputer=preprocessing["imputer"], scaler=preprocessing["scaler"],
                  lookback=LOOKBACK, future_horizons=HORIZONS)
    summary = {part: CausalWindowDataset(cell_ids=ids, **common)
               for part, ids in split.items()}
    waveform = {part: WaveformWindowDataset(cell_ids=ids,
                                            waveform_dir=WAVEFORMS, **common)
                for part, ids in split.items()}
    if any(summary[part].rows != waveform[part].rows for part in split):
        raise ValueError("TCN summary and waveform rows do not align")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "status": "XJTU Stage-A TCN development ablation, already-inspected cells",
        "split_source": str(BASE / "split.json"),
        "preprocessing_source": str(BASE / "source_preprocessing.joblib"),
        "selection_metric": "validation current MAPE + 0.2 mean future MAPE",
        "horizons_cycles": HORIZONS, "lookback": LOOKBACK,
        "seed": SEED, "device": str(device),
        "models": {},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, datasets, factory, physics in (
        ("summary_tcn_direct", summary, BatteryHealthTCN, False),
        ("summary_tcn_physics", summary, BatteryHealthTCN, True),
        ("waveform_tcn_direct", waveform, WaveformHealthTCN, False),
        ("waveform_tcn_physics", waveform, WaveformHealthTCN, True),
    ):
        model, training = train_neural(datasets["train"],
                                       datasets["validation"],
                                       physics=physics, device=device,
                                       model_factory=factory)
        torch.save({"state_dict": model.state_dict(),
                    "signal_width": datasets["train"][0]["signal"].shape[-1],
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
    result["validation_selected_tcn"] = min(result["models"], key=lambda name:
        evaluation_score(result["models"][name]["validation"]))
    prior = {}
    for source in (BASE, WAVE):
        path = source / "development_results.json"
        old = json.loads(path.read_text(encoding="utf-8"))
        for name, row in old["models"].items():
            prior[name] = {
                part: {"current_cell_macro_mape_percent":
                       row[part]["current_cell_macro_mape_percent"],
                       "future_cell_macro_mape_percent":
                       row[part]["future_cell_macro_mape_percent"]}
                for part in ("validation", "test")}
    result["prior_comparison"] = prior
    result["validation_selected_across_prior_and_tcn"] = min(
        {**prior, **result["models"]}, key=lambda name:
        evaluation_score(({**prior, **result["models"]})[name]["validation"]))
    (OUT / "development_results.json").write_text(json.dumps(result, indent=2),
                                                     encoding="utf-8")
    print("Validation-selected across all variants:",
          result["validation_selected_across_prior_and_tcn"], flush=True)


if __name__ == "__main__":
    main()
