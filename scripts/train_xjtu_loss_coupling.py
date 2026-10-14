"""Single-factor future-loss gradient ablation; preserve every fitted result.

Reuses the twelve exact joint-training timing/rate runs already completed.
Fits twelve matched detached-gradient runs (A/B, GRU/TCN, three seeds).
"""

from datetime import datetime, timezone
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import sys
import traceback

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from battery_smetimes.data.battery_dataset import DomainCellBalancedSampler
from battery_smetimes.data.cycle_time import CycleTimeWindowDataset
from battery_smetimes.models.cycle_time import CycleTimeHealthModel
from battery_smetimes.models.loss_coupling import LossSeparatedHealthModel
from battery_v2_features import load_cell, load_manifest
from scripts.train_mit_joint_trajectory import save_json, source_hash
from scripts.train_xjtu_cycle_time import (
    CODE, OUT as PREVIOUS, SEEDS, SOURCES, digest_cells, digest_files, evaluate_and_export,
)
from train_xjtu_stage_a import BATCH_SIZE, HORIZONS, LOOKBACK, evaluation_score, train_neural


OUT = Path("battery_models_v3/xjtu_loss_coupling")


def gradient_diagnostic(model, dataset, device):
    """Source-cell diagnostic of the two loss directions, not a selection rule."""
    # cuDNN RNN evaluation forwards do not retain backward intermediates.
    # Use a CPU copy: keep dropout disabled and never mutate saved fit weights.
    model = copy.deepcopy(model).cpu()
    device = torch.device("cpu")
    model.eval()
    sampler = DomainCellBalancedSampler(dataset, 8 * BATCH_SIZE, seed=20929)
    shared = [p for name, p in model.named_parameters()
              if "rate_head" not in name and "acceleration_head" not in name]
    rows = []
    for batch in DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE):
        output = CycleTimeHealthModel.forward(model, batch["signal"].to(device),
            batch["mask"].to(device), batch["metadata"].to(device), HORIZONS,
            cycle_context=batch["cycle_context"].to(device))
        weight = batch["sample_weight"].to(device)
        current = (F.huber_loss(output["current_soh"], batch["target_soh"].to(device),
                               delta=.03, reduction="none") * weight).sum() / weight.sum()
        available = batch["future_available"].to(device)
        target = torch.where(available, batch["future_soh"].to(device), output["future_soh"].detach())
        fw = available.float() * weight[:, None]
        future = (F.huber_loss(output["future_soh"], target, delta=.03,
                               reduction="none") * fw).sum() / fw.sum().clamp_min(1)
        a = torch.autograd.grad(current, shared, retain_graph=True, allow_unused=True)
        b = torch.autograd.grad(future, shared, allow_unused=True)
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        dot = sum((x * y).sum() for x, y in pairs)
        norm = torch.sqrt(sum(x.square().sum() for x, _ in pairs) * sum(y.square().sum() for _, y in pairs))
        rows.append(float((dot / norm.clamp_min(1e-12)).detach().cpu()))
    return {"training_batches": len(rows), "cosines": rows,
            "mean_cosine": float(np.mean(rows)), "opposing_fraction": float(np.mean(np.array(rows) < 0)),
            "scope": "fixed source-cell batches after training; diagnostic only, not test-based tuning"}


def save_comparison(folder, joint, separate, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, part in zip(axes, ("validation", "test")):
        x = np.arange(4)
        for offset, report, label, color in (
                (-.18, joint, "Joint loss", "#718096"),
                (.18, separate, "Separated gradients", "#258679")):
            values = [report[part]["current_cell_macro_mape_percent"]] + [
                report[part]["future_cell_macro_mape_percent"][str(h)] for h in HORIZONS]
            axis.bar(x + offset, values, width=.35, label=label, color=color)
        axis.set_xticks(x, ["Current", "+10", "+50", "+100"])
        axis.set_ylabel("Cell-macro MAPE (%)")
        axis.set_title(part.capitalize())
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle(name + " | identical inputs, parameters and training budget")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(folder / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, help="Resume an existing run; completed fits are reused")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = [row for row in load_manifest()["cells"] if row["dataset"] == "XJTU"]
    cells = {row["cell_id"]: load_cell(row) for row in records}
    by_id = {row["cell_id"]: row for row in records}
    plan = json.loads((PREVIOUS / "experiment_plan.json").read_text(encoding="utf-8"))
    if plan["identity"]["data_sha256"] != digest_cells(cells) or \
            plan["identity"]["code_sha256"] != digest_files(CODE):
        raise ValueError("joint reference data/code changed; matched comparison requires review")
    if plan["identity"]["torch_version"] != str(torch.__version__) or plan["identity"]["device"] != str(device):
        raise ValueError("runtime differs from joint references")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = args.resume or OUT / "runs" / run_id
    if args.resume:
        previous_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if previous_config["source_plan_sha256"] != source_hash(PREVIOUS / "experiment_plan.json"):
            raise ValueError("resume source plan changed")
        if previous_config["model_sha256"] != source_hash("battery_smetimes/models/loss_coupling.py"):
            raise ValueError("resume model changed")
        save_json(run / f"resume_{run_id}.json", {"script_sha256": source_hash(__file__),
            "reason": "Resume saved weights/results; diagnostic runs on CPU to allow evaluation-mode GRU backward"})
        run_id = previous_config["run_id"]
    else:
        run.mkdir(parents=True, exist_ok=False)
        save_json(run / "config.json", {"run_id": run_id, "stages": list(SOURCES), "backbones": ["gru", "tcn"],
        "seeds": list(SEEDS), "new_fits": 12, "reused_joint_fits": 12, "time_mode": "elapsed_rate",
        "source_plan": str(PREVIOUS / "experiment_plan.json"), "source_plan_sha256": source_hash(PREVIOUS / "experiment_plan.json"),
        "script_sha256": source_hash(__file__), "model_sha256": source_hash("battery_smetimes/models/loss_coupling.py"),
        "selection": "validation current MAPE + 0.2 mean future MAPE; same early stopping/budget as joint references",
        "changed_factor": "future-loss gradients cannot reach current SOH or shared encoder; physics heads still train"})
    def preserve_failure(kind, value, tb):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        save_json(run / f"failure_{stamp}.json", {"status": "failed", "error": str(value),
            "traceback": "".join(traceback.format_exception(kind, value, tb))})
        sys.__excepthook__(kind, value, tb)
    sys.excepthook = preserve_failure
    print("RUN", run.resolve(), flush=True)
    reports = []
    for stage, source in SOURCES.items():
        split = json.loads((source / "split.json").read_text(encoding="utf-8"))
        prep = joblib.load(source / "source_preprocessing.joblib")
        if prep["train_cell_ids"] != split["train"]:
            raise ValueError("preprocessor/cell split mismatch")
        datasets = {part: CycleTimeWindowDataset(cells, by_id, ids, prep["imputer"],
            scaler=prep["scaler"], lookback=LOOKBACK, future_horizons=HORIZONS,
            time_mode="elapsed_rate") for part, ids in split.items()}
        for backbone in ("gru", "tcn"):
            factory = partial(LossSeparatedHealthModel, context_width=datasets["train"].context_width,
                              backbone=backbone, separate_future_gradient=True)
            for seed in SEEDS:
                name = f"{stage}_{backbone}_{seed}"
                folder = run / name
                folder.mkdir(exist_ok=bool(args.resume))
                if (folder / "result.json").exists() and (folder / "comparison.png").exists():
                    saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                    if "source_gradient_diagnostic" in saved:
                        reports.append(saved)
                        print("REUSE", name, flush=True)
                        continue
                source_folder = PREVIOUS / "runs" / f"{stage}_{backbone}_elapsed_rate_{seed}"
                joint = json.loads((source_folder / "result.json").read_text(encoding="utf-8"))
                if joint["config"]["split"] != split:
                    raise ValueError("joint reference split differs")
                save_json(folder / "joint_reference.json", {"folder": str(source_folder),
                    "result_sha256": source_hash(source_folder / "result.json"),
                    "weights_sha256": source_hash(source_folder / "model.pt")})
                config = {"stage": stage, "backbone": backbone, "seed": seed, "split": split,
                          "input_mode": "elapsed_rate", "separate_future_gradient": True,
                          "preprocessing": str(source / "source_preprocessing.joblib")}
                save_json(folder / "config.json", config)
                print("START", name, flush=True)
                def progress(row):
                    with (folder / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    if row["epoch"] == 1 or row["epoch"] % 5 == 0:
                        print(name, "epoch", row["epoch"], "val-current",
                              round(row["validation_current_mape_percent"], 4), flush=True)
                if (folder / "model.pt").exists():
                    saved = torch.load(folder / "model.pt", map_location=device, weights_only=False)
                    if saved["config"] != config:
                        raise ValueError("resume checkpoint configuration changed")
                    model = factory(saved["signal_width"], physics_future=True).to(device)
                    model.load_state_dict(saved["state_dict"])
                    training = saved.get("training")
                    if training is None:
                        history = [json.loads(line) for line in (folder / "epochs.jsonl").read_text().splitlines()]
                        best_score, best_epoch = float("inf"), None
                        for row in history:
                            if row["validation_selection_score"] < best_score - 1e-5:
                                best_score, best_epoch = row["validation_selection_score"], row["epoch"]
                        training = {"best_epoch": best_epoch, "history": history,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "recovered_from_saved_epoch_log": True}
                    print("RELOAD COMPLETED WEIGHTS", name, flush=True)
                else:
                    model, training = train_neural(datasets["train"], datasets["validation"], physics=True,
                        device=device, model_factory=factory, seed=seed, epoch_callback=progress)
                    torch.save({"state_dict": model.state_dict(), "config": config, "training": training,
                        "signal_width": datasets["train"][0]["signal"].shape[-1],
                        "context_width": datasets["train"].context_width}, folder / "model.pt")
                result = {"config": config, "training": training}
                for part in ("validation", "test"):
                    result[part] = evaluate_and_export(model, datasets[part], device, folder / f"{part}.npz")
                result["joint_reference_metrics"] = {part: joint[part] for part in ("validation", "test")}
                result["joint_best_epoch"] = joint["training"]["best_epoch"]
                # Preserve useful training output before optional diagnostics/plotting.
                save_json(folder / "result.json", result)
                result["source_gradient_diagnostic"] = gradient_diagnostic(model, datasets["train"], device)
                save_json(folder / "result.json", result)
                save_comparison(folder, joint, result, name)
                reports.append(result)
                save_json(run / "progress.json", {"completed_fits": len(reports), "expected_fits": 12})
                print("SAVED", name, "test-current joint/detached",
                      round(joint["test"]["current_cell_macro_mape_percent"], 4),
                      round(result["test"]["current_cell_macro_mape_percent"], 4), flush=True)
    save_json(run / "results.json", {"run_id": run_id, "reports": reports,
        "status": "matched gradient ablation on already-inspected XJTU development cells"})
    save_json(OUT / "latest.json", {"run_id": run_id, "run_path": str(run)})
    print("COMPLETE", run.resolve(), flush=True)


if __name__ == "__main__":
    main()
