"""Small-model training with validation-only early stopping and frozen cell folds."""

import argparse
import copy
from datetime import datetime, timezone
import json
from importlib.metadata import version
import logging
import os
from pathlib import Path
import platform
import shutil
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import torch

from battery_smetimes.physical_types import load_registry, require_same_registered_family
from battery_smetimes.research_baselines import evaluate_baseline
from battery_smetimes.research_sequences import (HistoryPreprocessor, SmallForecaster,
    fit_epoch, predict_sequence, training_contexts)
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data
from scripts.train_soh_research_v5 import save_scores
from scripts.summarize_soh_research_v5 import summarize


def instantiate(processor, architecture, parameters, seed, device, llm_config=None):
    torch.manual_seed(seed)
    if llm_config is not None:
        from battery_smetimes.research_llm import FrozenSMETimesForecaster
        variant = llm_config["variants"][architecture]
        return FrozenSMETimesForecaster(llm_config["checkpoint"], 2 * processor.raw_width + 1,
            seed=seed, dropout=parameters["dropout"], device=device,
            patch_size=llm_config["patch_size"], max_text_tokens=llm_config["max_text_tokens"], **variant)
    return SmallForecaster(architecture, 2 * processor.raw_width + 1, processor.stats_width,
        hidden=parameters["hidden"], dropout=parameters["dropout"]).to(device)


def save_network(folder, bundle, parameters, epochs, history):
    if hasattr(bundle["network"], "backbone"):
        from battery_smetimes.research_llm import adapter_state
        state = adapter_state(bundle["network"])
    else:
        state = bundle["network"].state_dict()
    torch.save(state, folder / "model.pt")
    joblib.dump(bundle["processor"], folder / "preprocessor.joblib")
    save_json(folder / "model.json", {"architecture": bundle["model_name"], "parameters": parameters,
        "epochs": epochs, "physical_family": bundle["physical_family"], "registry": bundle["registry"],
        "train_ids": bundle["train_ids"], "origins": bundle["origins"],
        "history_width": 2 * bundle["processor"].raw_width + 1,
        "statistics_width": bundle["processor"].stats_width,
        "seed": bundle["seed"], "total_parameters": sum(p.numel() for p in bundle["network"].parameters()),
        "trainable_parameters": sum(p.numel() for p in bundle["network"].parameters() if p.requires_grad),
        "training_history": history})
    if hasattr(bundle["network"], "backbone"):
        save_json(folder / "llm.json", {"checkpoint": bundle["llm_config"]["checkpoint"],
            "checkpoint_files_sha256": bundle["checkpoint_files_sha256"],
            "variant": bundle["llm_config"]["variants"][bundle["model_name"]],
            "parameters": bundle["network"].parameter_counts(),
            "weights_file": "model.pt contains adapter/head only; frozen backbone identified by hashes"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/soh_research_v5_sequence_models.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold", action="append")
    parser.add_argument("--architecture", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-global-summary", action="store_true")
    args = parser.parse_args()
    neural_config = json.loads(args.config.read_text())
    llm_config = neural_config if neural_config.get("kind") == "frozen_llm" else None
    predictor, epoch_function = predict_sequence, fit_epoch
    if llm_config is not None:
        from battery_smetimes.research_llm import (attach_cards, fit_llm_epoch,
            predict_llm, adapter_state, load_adapter)
        predictor = predict_llm
        def epoch_function(network, contexts, optimizer, seed, epoch, batch_size, device):
            return fit_llm_epoch(network, contexts, optimizer, seed, epoch, batch_size, device,
                                 llm_config["balance_coefficient"])
    checkpoint_hashes = {}
    if llm_config is not None:
        checkpoint = Path(llm_config["checkpoint"])
        checkpoint_hashes = {str(p.relative_to(checkpoint)): sha(p) for p in checkpoint.iterdir()
                             if p.is_file() and p.suffix in (".json", ".safetensors", ".model")}
        if "model.safetensors" not in checkpoint_hashes:
            raise ValueError("missing local audited LLM weights")
    audit_folder = Path(neural_config["frozen_plan"])
    frozen = json.loads((audit_folder / "config.json").read_text())
    config = frozen["config"]
    audit = json.loads((audit_folder / "audit.json").read_text())
    registry_file = audit_folder / "source_snapshot/battery_physical_registry.json"
    if sha(registry_file) != frozen["registry_sha256"] or sha(config["integrity_audit"]) != frozen["integrity_sha256"]:
        raise ValueError("frozen specifications/integrity rules changed")
    registry = load_registry(registry_file)
    registered = json.loads((audit_folder / "cell_registry.json").read_text())["cells"]
    data = Path(audit["prepared_folder"])
    if sha(data / "manifest.json") != frozen["prepared_manifest_sha256"]:
        raise ValueError("frozen data manifest changed")
    manifest, cells = load_data(data, config["integrity_audit"])
    for key, cell in cells.items():
        cell.update({name: registered[key][name] for name in ("physical_family", "reference_established_source_cycle")})
    plans = json.loads((audit_folder / "splits.json").read_text())["plans"]
    fold_names = args.fold or (["XJTU_fold1"] if args.smoke else [p["name"] for p in plans])
    if set(fold_names) - {p["name"] for p in plans}:
        raise ValueError("unknown frozen fold")
    plans = [p for p in plans if p["name"] in fold_names]
    architectures = args.architecture or neural_config["architectures"]
    if set(architectures) - set(neural_config["architectures"]):
        raise ValueError("unknown architecture")
    seeds = args.seed or ([config["training_seeds"][0]] if args.smoke else config["training_seeds"])
    if set(seeds) - set(config["training_seeds"]):
        raise ValueError("unregistered training seed")
    candidates = neural_config["candidates"][:1] if args.smoke else neural_config["candidates"]
    max_epochs = neural_config["smoke_epochs"] if args.smoke else neural_config["max_epochs"]
    folder = Path("experiments/soh_research_v5/runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(folder / "training.log", encoding="utf-8"), logging.StreamHandler()])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = args.device
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    files = [Path(__file__), args.config, Path("battery_smetimes/research_sequences.py"),
        Path("battery_smetimes/research_baselines.py"), Path("battery_smetimes/physical_types.py"),
        Path("battery_smetimes/forecast.py"), Path("scripts/train_same_type_forecast.py"),
        Path("scripts/train_soh_research_v5.py"), Path("scripts/prepare_same_type_forecast.py"),
        Path("scripts/summarize_soh_research_v5.py")]
    if llm_config is not None:
        files.append(Path("battery_smetimes/research_llm.py"))
    for file in files:
        target = folder / "source_snapshot" / (file.name if file.is_absolute() else file)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    shutil.copytree(audit_folder, folder / "frozen_audit")
    save_json(folder / "config.json", {"config": config, "sequence_config": neural_config, "folds": fold_names,
        "seeds": seeds, "architectures": architectures, "smoke": args.smoke, "device": device,
        "python": platform.python_version(), "torch": torch.__version__,
        "packages": {key: version(key) for key in ("numpy", "scipy", "scikit-learn", "transformers", "joblib", "psutil", "matplotlib")},
        "cuda_device_name": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "checkpoint_files_sha256": checkpoint_hashes,
        "source_sha256": {str(p): sha(p) for p in files},
        "prepared_files_sha256": {r["cell_id"]: r["sha256"] for r in manifest["cells"]},
        "evidence_level": "two-epoch single-fold engineering smoke" if args.smoke else "public development grouped CV"})
    save_json(folder / "splits.json", {"plans": plans})
    progress = {"complete": False, "smoke": args.smoke, "completed": [], "summary": []}
    save_json(folder / "progress.json", progress)
    save_json(Path("experiments/soh_research_v5/latest_sequence_run.json"), {"folder": str(folder.resolve())})
    try:
        for plan in plans:
            family = registered[plan["train"][0]]["physical_family"]
            require_same_registered_family(family, [cells[k]["physical_family"] for k in plan["train"]], registry)
            for seed in seeds:
                for architecture in architectures:
                    destination = folder / plan["name"] / str(seed) / architecture
                    destination.mkdir(parents=True, exist_ok=False)
                    save_json(destination / "config.json", {"plan": plan, "seed": seed, "model": architecture,
                        "candidates": candidates, "smoke": args.smoke, "physical_family": family})
                    processor = HistoryPreprocessor().fit(cells, plan["train"], config["origins_source_cycles"])
                    contexts = training_contexts(cells, plan["train"], processor, neural_config["max_future_targets_per_context"])
                    scores = []
                    for index, parameters in enumerate(candidates):
                        candidate_dir = destination / f"candidate_{index}"
                        candidate_dir.mkdir()
                        candidate_started = time.perf_counter()
                        network = instantiate(processor, architecture, parameters, seed, device, llm_config)
                        optimizer = torch.optim.AdamW((p for p in network.parameters() if p.requires_grad), lr=parameters["learning_rate"],
                                                      weight_decay=parameters["weight_decay"])
                        bundle = {"network": network, "processor": processor, "physical_family": family,
                            "registry": registry, "device": device, "model_name": architecture,
                            "train_ids": plan["train"], "origins": config["origins_source_cycles"], "seed": seed}
                        if llm_config is not None:
                            bundle.update(llm_config=llm_config, checkpoint_files_sha256=checkpoint_hashes)
                            attach_cards(bundle, contexts, cells)
                        best, best_state, best_epoch, bad, history = float("inf"), None, 0, 0, []
                        for epoch in range(1, max_epochs + 1):
                            loss = epoch_function(network, contexts, optimizer, seed, epoch, neural_config["batch_contexts"], device)
                            row = {"epoch": epoch, "train_mse": loss}
                            row.update(getattr(network, "last_epoch_diagnostics", {}))
                            if epoch % neural_config["validation_every_epochs"] == 0 or epoch == max_epochs:
                                report, arrays = evaluate_baseline(bundle, cells, plan["validation"], predictor=predictor)
                                value = report["summary"]["100"]["cell_macro_mape_percent"]
                                row["validation_cell_macro_mape_percent_origin100"] = value
                                if value is None or not np.isfinite(value):
                                    raise ValueError("no finite primary validation error")
                                if value < best:
                                    best, best_epoch, bad = value, epoch, 0
                                    best_state = adapter_state(network) if llm_config is not None else {
                                        k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
                                else:
                                    bad += 1
                            history.append(row)
                            save_json(candidate_dir / "training_history.json", history)
                            logging.info("EPOCH %s %s seed=%s candidate=%s epoch=%s train=%s val=%s", plan["name"],
                                architecture, seed, index, epoch, loss, row.get("validation_cell_macro_mape_percent_origin100"))
                            if bad >= neural_config["patience_validations"]:
                                break
                        if llm_config is not None:
                            load_adapter(network, best_state)
                        else:
                            network.load_state_dict(best_state)
                        save_network(candidate_dir, bundle, parameters, best_epoch, history)
                        report, arrays = evaluate_baseline(bundle, cells, plan["validation"], predictor=predictor)
                        save_scores(candidate_dir / "validation", report, arrays)
                        save_json(candidate_dir / "resources.json", {"total_candidate_seconds_including_model_load_cards_training_and_validation":
                            time.perf_counter() - candidate_started, "device": device,
                            "cuda_peak_allocated_mb_cumulative_process": torch.cuda.max_memory_allocated() / 2 ** 20 if device.startswith("cuda") else None})
                        scores.append({"index": index, "parameters": parameters, "validation_mape_percent": best,
                                       "selected_epochs": best_epoch})
                        save_json(destination / "validation_scores.json", scores)
                        del optimizer, bundle, network, best_state
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()
                    selected = min(scores, key=lambda r: r["validation_mape_percent"])
                    save_json(destination / "selected.json", selected)
                    ids = plan["train"] + plan["validation"]
                    if set(ids) & set(plan["test"]):
                        raise ValueError("test cell entered refit")
                    refit_pipeline_started = time.perf_counter()
                    processor = HistoryPreprocessor().fit(cells, ids, config["origins_source_cycles"])
                    contexts = training_contexts(cells, ids, processor, neural_config["max_future_targets_per_context"])
                    network = instantiate(processor, architecture, selected["parameters"], seed, device, llm_config)
                    optimizer = torch.optim.AdamW((p for p in network.parameters() if p.requires_grad), lr=selected["parameters"]["learning_rate"],
                                                  weight_decay=selected["parameters"]["weight_decay"])
                    bundle = {"network": network, "processor": processor, "physical_family": family,
                        "registry": registry, "device": device, "model_name": architecture,
                        "train_ids": ids, "origins": config["origins_source_cycles"], "seed": seed}
                    if llm_config is not None:
                        bundle.update(llm_config=llm_config, checkpoint_files_sha256=checkpoint_hashes)
                        attach_cards(bundle, contexts, cells)
                    started, history = time.perf_counter(), []
                    for epoch in range(1, selected["selected_epochs"] + 1):
                        loss = epoch_function(network, contexts, optimizer, seed, epoch, neural_config["batch_contexts"], device)
                        history.append({"epoch": epoch, "train_mse": loss, **getattr(network, "last_epoch_diagnostics", {})})
                        save_json(destination / "refit_history.json", history)
                    save_network(destination, bundle, selected["parameters"], selected["selected_epochs"], history)
                    report, arrays = evaluate_baseline(bundle, cells, plan["test"], predictor=predictor)
                    save_scores(destination / "test", report, arrays)
                    import psutil
                    save_json(destination / "resources.json", {"refit_and_test_seconds": time.perf_counter() - started,
                        "full_refit_pipeline_and_test_seconds_including_preprocessing_model_load_card_cache": time.perf_counter() - refit_pipeline_started,
                        "process_rss_mb_at_end_not_peak": psutil.Process().memory_info().rss / 2 ** 20,
                        "cuda_peak_allocated_mb_cumulative_run": torch.cuda.max_memory_allocated() / 2 ** 20 if device.startswith("cuda") else None,
                        "device": device, "parameters": sum(p.numel() for p in network.parameters()),
                        "training_cells": len(ids), "training_contexts": len(contexts),
                        "model_bytes": (destination / "model.pt").stat().st_size})
                    progress["completed"].append({"fold": plan["name"], "seed": seed, "model": architecture})
                    for origin, row in report["summary"].items():
                        progress["summary"].append({"fold": plan["name"], "seed": seed, "model": architecture,
                                                    "origin": int(origin), **row})
                    save_json(folder / "progress.json", progress)
                    if not args.no_global_summary:
                        summarize()
                    logging.info("SAVED %s %s seed=%s origin100=%s", plan["name"], architecture, seed, report["summary"]["100"])
                    del optimizer, bundle, network, contexts
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
        progress["complete"] = True
        save_json(folder / "summary.json", progress)
        save_json(folder / "progress.json", progress)
        if not args.no_global_summary:
            summarize()
        logging.info("COMPLETE SEQUENCE RUN %s", folder.resolve())
    except BaseException:
        save_json(folder / "failure.json", {"traceback": traceback.format_exc(), "completed": progress["completed"]})
        raise


if __name__ == "__main__":
    main()
