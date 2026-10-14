"""Saved source-validation/fixed-refit units for shared research neural transfer.

These are worker primitives, NOT an ungated public-data launcher. The future
dispatcher must use the existing main-completion/numerical gates, all source
CV folds and five seeds. Target labels never set epochs or parameters here.
"""

import json
import logging
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
import psutil
import torch

from battery_smetimes.research_sequences import fit_epoch
from battery_smetimes.research_llm import adapter_state, load_adapter, fit_llm_epoch
from battery_smetimes.research_transfer_baselines import evaluate_transfer
from battery_smetimes.research_transfer_sequences import (
    TransferHistoryPreprocessor, make_transfer_sequence_bundle, predict_transfer_sequence,
    transfer_training_contexts, _source_ids,
)
from battery_smetimes.research_transfer_signals import RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.prepare_same_type_forecast import save_json


def _state(network):
    return adapter_state(network) if hasattr(network, "backbone") else {
        k: v.detach().cpu().clone() for k, v in network.state_dict().items()}


def _restore(network, state):
    if hasattr(network, "backbone"):
        load_adapter(network, state)
    else:
        network.load_state_dict(state)


def _save_scores(folder, report, arrays):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions.npz", **arrays)
    save_json(folder / "metrics.json", report)


def save_shared_neural_checkpoint(folder, bundle, epochs, history, *, checkpoint_hashes=None):
    """Trusted local weights only; LLM export requires original backbone hashes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    network = bundle["network"]
    is_llm = hasattr(network, "backbone")
    if is_llm and (not checkpoint_hashes or "model.safetensors" not in checkpoint_hashes):
        raise ValueError("shared LLM export needs the frozen original checkpoint fingerprints")
    torch.save(_state(network), folder / "model.pt")
    joblib.dump(bundle["processor"], folder / "preprocessor.joblib")
    save_json(folder / "model.json", {key: bundle[key] for key in (
        "schema", "mode", "not_a_production_route", "plan", "allowed_cells", "input_schema", "source_train_ids",
        "source_only_preprocessor_fit_ids", "origins", "model_name", "parameters", "seed", "device", "llm_config")})
    info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    info.update(epochs=epochs, training_history=history, torch=torch.__version__,
        history_width=2 * bundle["processor"].raw_width + 1, statistics_width=bundle["processor"].stats_width,
        total_parameters=sum(p.numel() for p in network.parameters()),
        trainable_parameters=sum(p.numel() for p in network.parameters() if p.requires_grad),
        checkpoint_files_sha256=checkpoint_hashes if is_llm else {})
    save_json(folder / "model.json", info)
    save_json(folder / "checkpoint_manifest.json", {"schema": 1, "trusted_local_only": True,
        "artifact_sha256": {name: stream_sha(folder / name) for name in ("model.pt", "model.json", "preprocessor.joblib")},
        "scope": "weight/scaler fingerprint prerequisite; parent source/configuration/static numeric audit and executable replay remain separate"})


def load_shared_neural_checkpoint(folder, *, device=None):
    """No fit; callers must first verify their frozen run's source/configuration.

    Does not certify an entire source-CV run or arbitrary uploaded joblib files.
    The manifest identifies bytes, not a cryptographic author/authenticity.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    required = {"model.pt", "model.json", "preprocessor.joblib"}
    if manifest["schema"] != 1 or set(manifest["artifact_sha256"]) != required or not manifest["trusted_local_only"]:
        raise ValueError("incomplete shared neural checkpoint fingerprint manifest")
    for name, digest in manifest["artifact_sha256"].items():
        if stream_sha(folder / name) != digest:
            raise ValueError("shared neural weights/processor/metadata changed")
    info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if info["mode"] != RESEARCH_MODE or not info["not_a_production_route"]:
        raise ValueError("checkpoint is not an explicit research transfer model")
    _source_ids(info["source_train_ids"], info["plan"])
    if info["source_only_preprocessor_fit_ids"] != info["source_train_ids"]:
        raise ValueError("checkpoint preprocessing IDs disagree with source fitting")
    if info["torch"] != torch.__version__:
        raise ValueError("Torch version differs; exact shared checkpoint replay is not certified")
    for name, digest in info["checkpoint_files_sha256"].items():
        if stream_sha(Path(info["llm_config"]["checkpoint"]) / name) != digest:
            raise ValueError("shared LLM original checkpoint changed")
    if info["model_name"].startswith("llm_") and "model.safetensors" not in info["checkpoint_files_sha256"]:
        raise ValueError("shared LLM original checkpoint fingerprint missing")
    processor = joblib.load(folder / "preprocessor.joblib")
    if any(getattr(processor, attribute) != info[key] for attribute, key in (
            ("train_ids", "source_train_ids"), ("origins", "origins"), ("plan", "plan"),
            ("input_schema", "input_schema"), ("allowed_cells", "allowed_cells"))):
        raise ValueError("saved shared processor identities disagree with model metadata")
    if 2 * processor.raw_width + 1 != info["history_width"] or processor.stats_width != info["statistics_width"]:
        raise ValueError("saved shared neural feature dimensions differ")
    bundle = make_transfer_sequence_bundle(processor, info["model_name"], info["parameters"], info["seed"],
        info["device"] if device is None else device, llm_config=info["llm_config"])
    _restore(bundle["network"], torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    bundle["network"].eval()
    return bundle, {"no_fit": True, "artifact_sha256": manifest["artifact_sha256"],
        "scope": "trusted local single shared checkpoint reconstruction only; parent provenance and predictive replay are separate"}


def train_shared_source_unit(cells, plan, origins, architecture, parameters, seed, device, config, destination,
                             *, source_fold=None, fixed_refit_epochs=None, llm_config=None, checkpoint_hashes=None):
    """Train one source-only fold OR refit all source cells for fixed epochs.

    No target evaluation is called during training. For source CV the full
    declared validation membership is required at N100, including hard cells.
    Refit epochs must be selected by the outer dispatcher using SOURCE folds.
    Existing attempts are not overwritten or automatically restarted.
    """
    if (source_fold is None) == (fixed_refit_epochs is None):
        raise ValueError("choose exactly source validation or fixed source refit")
    if source_fold is not None:
        if source_fold not in plan["source_selection_folds"]:
            raise ValueError("source fold differs from unchanged frozen transfer plan")
        ids, validation = list(source_fold["source_train"]), list(source_fold["source_validation"])
        if (not validation or len(validation) != len(set(validation)) or set(ids) & set(validation)
                or set(ids + validation) != set(plan["source_refit"])):
            raise ValueError("source-only validation membership incomplete or overlapping")
        _source_ids(validation, plan)
        max_epochs = config["max_epochs"]
    else:
        ids, validation, max_epochs = list(plan["source_refit"]), [], fixed_refit_epochs
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs < 1 or max_epochs > config["max_epochs"]:
        raise ValueError("invalid source-only training epoch budget")
    for field in ("validation_every_epochs", "patience_validations", "batch_contexts", "max_future_targets_per_context"):
        if isinstance(config[field], bool) or not isinstance(config[field], int) or config[field] < 1:
            raise ValueError("invalid source-only training configuration")
    _source_ids(ids, plan)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "config.json", {"plan": plan, "origins": origins, "model": architecture, "parameters": parameters,
        "seed": seed, "device": device, "configuration": config, "source_fold": source_fold,
        "fixed_refit_epochs": fixed_refit_epochs, "selection_uses_target_labels": False})
    history, started = [], time.perf_counter()
    try:
        if architecture.startswith("llm_"):
            if llm_config is None or not checkpoint_hashes or "model.safetensors" not in checkpoint_hashes:
                raise ValueError("source LLM fitting requires the frozen original checkpoint fingerprints")
            checkpoint = Path(llm_config["checkpoint"]).resolve()
            for name, digest in checkpoint_hashes.items():
                path = checkpoint / name
                if not path.resolve().is_relative_to(checkpoint) or stream_sha(path) != digest:
                    raise ValueError("source LLM checkpoint bytes changed before fitting")
        processor = TransferHistoryPreprocessor().fit(cells, ids, plan, origins)
        bundle = make_transfer_sequence_bundle(processor, architecture, parameters, seed, device, llm_config=llm_config)
        contexts = transfer_training_contexts(bundle, cells, config["max_future_targets_per_context"])
        network = bundle["network"]
        optimizer = torch.optim.AdamW((p for p in network.parameters() if p.requires_grad),
            lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
        best, best_epoch, best_state, bad = float("inf"), 0, None, 0
        for epoch in range(1, max_epochs+1):
            if architecture.startswith("llm_"):
                loss = fit_llm_epoch(network, contexts, optimizer, seed, epoch, config["batch_contexts"], device,
                                     llm_config["balance_coefficient"])
            else:
                loss = fit_epoch(network, contexts, optimizer, seed, epoch, config["batch_contexts"], device)
            row = {"epoch": epoch, "train_mse": loss, **getattr(network, "last_epoch_diagnostics", {})}
            if validation and (epoch % config["validation_every_epochs"] == 0 or epoch == max_epochs):
                report, arrays = evaluate_transfer(bundle, cells, validation, predictor=predict_transfer_sequence)
                primary = [r for r in report["by_cell"] if r["origin"] == 100]
                if len(primary) != len(validation) or {r["cell_id"] for r in primary} != set(validation):
                    raise ValueError("source validation must retain all required primary cells")
                value = report["summary"]["100"]["cell_macro_mape_percent"]
                if value is None or not np.isfinite(value):
                    raise ValueError("source validation error unavailable or nonfinite")
                row["source_validation_mape_percent_origin100"] = value
                if value < best:
                    best, best_epoch, best_state, bad = value, epoch, _state(network), 0
                    _save_scores(destination / "best_source_validation", report, arrays)
                else:
                    bad += 1
            history.append(row)
            save_json(destination / "training_history.json", history)
            logging.info("SHARED SOURCE EPOCH %s %s seed=%s epoch=%s source_val=%s", plan.get("name"), architecture, seed,
                         epoch, row.get("source_validation_mape_percent_origin100"))
            if validation and bad >= config["patience_validations"]:
                break
        if validation:
            if best_state is None:
                raise ValueError("source-fold candidate has no selected checkpoint")
            _restore(network, best_state)
            report, arrays = evaluate_transfer(bundle, cells, validation, predictor=predict_transfer_sequence)
            _save_scores(destination / "source_validation", report, arrays)
        else:
            best_epoch, report = max_epochs, None
        save_shared_neural_checkpoint(destination, bundle, best_epoch, history, checkpoint_hashes=checkpoint_hashes)
        resources = {"pipeline_seconds_including_preprocessing_backbone_cards_training_validation": time.perf_counter()-started,
            "process_rss_mb_at_end_not_peak": psutil.Process().memory_info().rss/2**20,
            "cuda_peak_allocated_mb_cumulative_process": torch.cuda.max_memory_allocated()/2**20 if device.startswith("cuda") else None,
            "device": device, "source_train_ids": ids, "source_validation_ids": validation,
            "source_only_preprocessor_fit_ids": processor.train_ids, "training_contexts": len(contexts),
            "total_parameters": sum(p.numel() for p in network.parameters()),
            "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
            "model_bytes": (destination / "model.pt").stat().st_size, "not_an_idle_latency_benchmark": True}
        save_json(destination / "resources.json", resources)
        result = {"complete": True, "selected_epochs": best_epoch,
            "source_validation_mape_percent_origin100": best if validation else None,
            "source_validation_by_cell": report["by_cell"] if validation else [],
            "target_evaluated_during_training": False}
        save_json(destination / "report.json", result)
        return bundle, result
    except BaseException:
        save_json(destination / "failure.json", {"traceback": traceback.format_exc(), "training_history": history,
            "target_evaluated_during_training": False})
        raise
