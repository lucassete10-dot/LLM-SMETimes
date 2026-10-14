"""Three-arm, three-seed XJTU timing experiment on frozen A/B cell splits.

Run from the repository root with `python -m scripts.train_xjtu_cycle_time`.
Every arm has identical parameters and initial function; the only change is
which causal timing channels are revealed. Completed runs are fingerprinted
and reused. A missing run is trained without replacing legacy artifacts.
"""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from battery_smetimes.data.cycle_time import CycleTimeWindowDataset, MODES
from battery_smetimes.models.cycle_time import CycleTimeHealthModel
from battery_v2_features import load_cell, load_manifest
from train_xjtu_stage_a import (
    BATCH_SIZE, HORIZONS, LOOKBACK, MAX_EPOCHS, PATIENCE,
    evaluation_score, metric_rows, train_neural,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "battery_models_v3/xjtu_cycle_time_ablation"
SEEDS = (2026, 2027, 2028)
SOURCES = {
    "A": ROOT / "battery_models_v3/xjtu_stage_a",
    "B": ROOT / "battery_models_v3/xjtu_stage_b_protocol_holdout",
}
CODE = (
    "battery_smetimes/data/cycle_time.py", "battery_smetimes/models/cycle_time.py",
    "battery_smetimes/data/battery_dataset.py", "battery_smetimes/models/temporal.py",
    "train_xjtu_stage_a.py", "scripts/train_xjtu_cycle_time.py",
)


def digest_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def digest_cells(cells):
    digest = hashlib.sha256()
    for key, cell in sorted(cells.items()):
        digest.update(key.encode())
        for array in (cell.cycle, cell.matrix("signal_history"), cell.soh,
                      cell.soh_available, cell.capacity_review_flag):
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def evaluate_and_export(model, dataset, device, path):
    rows, ids, cycles = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=256, shuffle=False):
            output = model(batch["signal"].to(device), batch["mask"].to(device),
                           batch["metadata"].to(device), HORIZONS,
                           cycle_context=batch["cycle_context"].to(device))
            current = output["current_soh"].cpu().numpy()
            future = output["future_soh"].cpu().numpy()
            for i, key in enumerate(batch["cell_id"]):
                ids.append(key)
                cycles.append(int(batch["target_cycle"][i]))
                rows.append({"cell_id": key, "target": float(batch["target_soh"][i]),
                             "estimate": float(current[i]),
                             "future_target": batch["future_soh"][i].numpy().tolist(),
                             "future_estimate": future[i].tolist()})
    np.savez_compressed(
        path, cell_id=np.asarray(ids), cycle=np.asarray(cycles),
        target=np.asarray([r["target"] for r in rows]),
        prediction=np.asarray([r["estimate"] for r in rows]),
        future_target=np.asarray([r["future_target"] for r in rows]),
        future_prediction=np.asarray([r["future_estimate"] for r in rows]),
        horizons=np.asarray(HORIZONS))
    return metric_rows(rows)


def summarize(output):
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((output / "runs").glob("*/result.json"))]
    summary = {"status": "development; seeds measure training stability on fixed inspected cells",
               "expected_runs": 36, "completed_runs": len(reports), "splits": {}}
    for stage in SOURCES:
        candidates = {}
        for backbone in ("gru", "tcn"):
            for mode in MODES:
                runs = [r for r in reports if r["config"]["stage"] == stage and
                        r["config"]["backbone"] == backbone and
                        r["config"]["time_mode"] == mode]
                if not runs:
                    continue
                aggregate = {"seeds": [r["config"]["seed"] for r in runs],
                             "complete": len(runs) == len(SEEDS)}
                for part in ("validation", "test"):
                    current = [r[part]["current_cell_macro_mape_percent"] for r in runs]
                    aggregate[part] = {
                        "current_mape_mean": float(np.mean(current)),
                        "current_mape_sd": float(np.std(current)),
                        "current_mape_each_seed": current,
                        "future_mape_mean": {str(h): float(np.mean([
                            r[part]["future_cell_macro_mape_percent"][str(h)]
                            for r in runs])) for h in HORIZONS},
                        "selection_score_mean": float(np.mean([
                            evaluation_score(r[part]) for r in runs])),
                    }
                candidates[f"{backbone}_{mode}"] = aggregate
        complete = {name: row for name, row in candidates.items() if row["complete"]}
        summary["splits"][stage] = {
            "candidates": candidates,
            "selected_on_validation": min(complete, key=lambda n:
                complete[n]["validation"]["selection_score_mean"]) if len(complete) == 6 else None,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    parser.add_argument("--backbones", nargs="+", choices=("gru", "tcn"), default=["gru", "tcn"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = [r for r in load_manifest()["cells"] if r["dataset"] == "XJTU"]
    cells = {r["cell_id"]: load_cell(r) for r in records}
    by_id = {r["cell_id"]: r for r in records}
    identity = {
        "code_sha256": digest_files([ROOT / name for name in CODE]),
        "data_sha256": digest_cells(cells), "torch_version": str(torch.__version__),
        "device": str(device), "threads": 4,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
        "horizons": list(HORIZONS), "lookback": LOOKBACK,
        "selection": "validation current MAPE + 0.2 mean future MAPE",
        "context_scaling": "relative cycles / 100; previous gap / 10; trailing5 OLS raw-feature slope per cycle * 100 clipped +/-8",
        "parameter_control": "identical 48-to-64 bias-free zero-initialized timing projection in every arm; baseline receives zeros",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"identity": identity, "stages": list(SOURCES), "backbones": ["gru", "tcn"],
            "modes": list(MODES), "seeds": list(SEEDS), "expected_runs": 36,
            "claim_boundary": "fixed inspected XJTU development cells; not blind acceptance; no EOL/RUL evaluation"}
    plan_path = args.output / "experiment_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("experiment identity changed; choose a new output directory")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print("PLAN", device, "36 runs; requested subset", args.stages, args.backbones,
          args.modes, args.seeds, flush=True)
    for stage in args.stages:
        source = SOURCES[stage]
        split = json.loads((source / "split.json").read_text(encoding="utf-8"))
        preprocessing = joblib.load(source / "source_preprocessing.joblib")
        if preprocessing["train_cell_ids"] != split["train"]:
            raise ValueError("frozen training/preprocessing identities differ")
        if sum(map(len, split.values())) != len(set().union(*map(set, split.values()))):
            raise ValueError("cell split overlaps")
        prep_hash = digest_files([source / "source_preprocessing.joblib"])
        for backbone in args.backbones:
            for seed in args.seeds:
                for mode in args.modes:
                    name = f"{stage}_{backbone}_{mode}_{seed}"
                    folder = args.output / "runs" / name
                    config = {**identity, "stage": stage, "backbone": backbone,
                              "seed": seed, "time_mode": mode, "split": split,
                              "preprocessing_source": str(source / "source_preprocessing.joblib"),
                              "preprocessing_sha256": prep_hash}
                    result_path = folder / "result.json"
                    if result_path.exists():
                        previous = json.loads(result_path.read_text(encoding="utf-8"))
                        if previous["config"] != config or not all(
                                (folder / n).exists() for n in ("model.pt", "validation.npz", "test.npz")):
                            raise ValueError(f"completed run identity/artifacts mismatch: {name}")
                        print("REUSE", name, flush=True)
                        continue
                    datasets = {part: CycleTimeWindowDataset(
                        cells, by_id, ids, preprocessing["imputer"],
                        scaler=preprocessing["scaler"], lookback=LOOKBACK,
                        future_horizons=HORIZONS, time_mode=mode)
                        for part, ids in split.items()}
                    folder.mkdir(parents=True, exist_ok=True)
                    print("START", name, flush=True)
                    started = time.monotonic()

                    def progress(row):
                        if row["epoch"] == 1 or row["epoch"] % 5 == 0:
                            print(name, "epoch", row["epoch"], "val-current",
                                  round(row["validation_current_mape_percent"], 4),
                                  "val-score", round(row["validation_selection_score"], 4),
                                  flush=True)

                    factory = partial(CycleTimeHealthModel,
                                      context_width=datasets["train"].context_width,
                                      backbone=backbone)
                    model, training = train_neural(
                        datasets["train"], datasets["validation"], physics=True,
                        device=device, model_factory=factory, seed=seed,
                        epoch_callback=progress)
                    torch.save({"state_dict": model.state_dict(), "config": config,
                                "signal_width": datasets["train"][0]["signal"].shape[-1],
                                "context_width": datasets["train"].context_width,
                                "physics_future": True}, folder / "model.pt")
                    result = {"config": config, "training": training}
                    for part in ("validation", "test"):
                        result[part] = evaluate_and_export(model, datasets[part], device,
                                                           folder / f"{part}.npz")
                    result["elapsed_seconds"] = time.monotonic() - started
                    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    summarize(args.output)
                    print("DONE", name, "test-current",
                          round(result["test"]["current_cell_macro_mape_percent"], 4),
                          "test-future", result["test"]["future_cell_macro_mape_percent"],
                          "seconds", round(result["elapsed_seconds"], 1), flush=True)
    summary = summarize(args.output)
    print("COMPLETE", summary["completed_runs"], "/", summary["expected_runs"], flush=True)


if __name__ == "__main__":
    main()
