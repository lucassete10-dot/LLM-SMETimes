"""Executable shared-checkpoint replay, not a replacement for static audits.

Only trusted local finalized research runs are accepted. The CLI installs their
frozen package paths in a fresh interpreter before importing this module.
"""

import importlib.metadata
import inspect
from pathlib import Path


from battery_smetimes.research_inference import verify_source_snapshot
from battery_smetimes.research_phase_gate import assert_main_complete, assert_main_numerical_audits
from battery_smetimes.research_transfer_neural_audit import read, audit_shared_neural_run, audit_shared_neural_unit
from battery_smetimes.research_transfer_neural_training import load_shared_neural_checkpoint
from battery_smetimes.research_transfer_neural_recovery import verify_root_utility_resolution
from battery_smetimes.research_transfer_sequences import (
    predict_transfer_sequence, make_transfer_sequence_bundle, TransferHistoryPreprocessor,
)
from battery_smetimes.research_sequences import SmallForecaster
from battery_smetimes.research_llm import FrozenSMETimesForecaster
from battery_smetimes.research_transfer_signals import RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import stream_sha
from battery_smetimes.research_transfer_vi120 import SCHEMA as STRICT_RAW_SCHEMA
from battery_smetimes.research_transfer_replay_checks import prefix_only, replay_predictions, causal_predictions
from scripts.train_soh_research_transfer_v5 import load_shared_cells


def verify_replay_run(run, *, progress_callback=None):
    """Recheck raw data and complete source/target numerical grid BEFORE load."""
    run = Path(run).resolve()
    config = verify_source_snapshot(run)
    state = read(run / "progress.json")
    if (config["prepare_only"] or not state["complete"] or not state["model_training_started"]
            or (run / "failure.json").exists() or (run / "launcher_failure.json").exists()):
        raise ValueError("shared replay requires successful completed real fitting, not prepared/partial/failed runs")
    for package, version in config["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("shared replay package version differs from original execution")
    prepared = Path(config["prepared"])
    for name, field in (("config.json", "prepared_config_sha256"), ("manifest.json", "prepared_manifest_sha256"),
                        ("splits.json", "prepared_splits_sha256")):
        if stream_sha(prepared / name) != config[field]:
            raise ValueError("shared replay prepared input contract changed")
    completion = assert_main_complete(config["main_run_bindings"])
    numerical = assert_main_numerical_audits(config["main_run_bindings"], run.parents[1] / "artifact_audits", completion=completion)
    root_paths = verify_root_utility_resolution(run, config)
    cells = load_shared_cells(prepared, progress_callback=progress_callback)
    static = audit_shared_neural_run(run, cells)
    if read(run / "static_run_audit.json") != static:
        raise ValueError("shared replay complete-run static evidence changed")
    return {"run": run, "config": config, "plans": read(run / "splits.json")["directions"], "cells": cells,
        "evidence": {"run": str(run), "run_config_sha256": stream_sha(run / "config.json"),
            "progress_sha256": stream_sha(run / "progress.json"), "splits_sha256": stream_sha(run / "splits.json"),
            "static_run_audit_sha256": stream_sha(run / "static_run_audit.json"),
            "prepared_manifest_sha256": config["prepared_manifest_sha256"], "source_sha256": config["source_sha256"],
            "main_completion": completion, "main_numerical": numerical, "static_finalized_units": static["finalized_units"],
            "root_utility_paths": root_paths,
            "shared_inputs_rederived": True,
            "strict_raw_signals_and_fingerprints_rechecked": config.get("input_schema") == STRICT_RAW_SCHEMA,
            "input_rederivation_scope": "raw MAT/HDF5 strict VI signals" if config.get("input_schema") == STRICT_RAW_SCHEMA else "shared signals from audited native prepared features; not a fresh raw MAT extraction",
            "weights_loaded": False, "training_performed": False,
            "not_same_type_deployment": True, "target_calibration": []}}


def verify_model_resolution(run, config):
    """Reject current-workspace/cached model classes before deserializing."""
    snapshot = Path(run) / "source_snapshot"
    objects = {"checkpoint_loader": load_shared_neural_checkpoint, "predictor": predict_transfer_sequence,
        "bundle_factory": make_transfer_sequence_bundle, "processor": TransferHistoryPreprocessor,
        "small_network": SmallForecaster, "llm_network": FrozenSMETimesForecaster,
        "replay_checks": replay_predictions, "prefix_removal": prefix_only, "causal_control": causal_predictions}
    paths = {}
    for name, value in objects.items():
        path = Path(inspect.getfile(value)).resolve()
        if not path.is_relative_to(snapshot.resolve()):
            raise ValueError("shared model implementation resolved outside frozen source; use a fresh saved replay interpreter")
        relative = path.relative_to(snapshot).as_posix()
        if config["source_sha256"].get(relative) != stream_sha(path):
            raise ValueError("shared predictor/class source is not in the verified snapshot")
        paths[name] = str(path)
    return paths


def load_verified_unit(verified, plan, seed, model, *, device="auto"):
    """No fit. Recheck parent and unit evidence immediately before weight load."""
    run, config, evidence = verified["run"], verified["config"], verified["evidence"]
    if (plan not in verified["plans"] or seed not in config["seeds"] or model not in config["models"]):
        raise ValueError("shared replay unit outside frozen direction/seed/model grid")
    for name, key in (("config.json", "run_config_sha256"), ("progress.json", "progress_sha256"),
                      ("splits.json", "splits_sha256"), ("static_run_audit.json", "static_run_audit_sha256")):
        if stream_sha(run / name) != evidence[key]:
            raise ValueError("verified shared parent evidence changed before deserialization")
    verify_source_snapshot(run)
    source_paths = verify_model_resolution(run, config)
    unit = run / plan["name"] / str(seed) / model
    current = audit_shared_neural_unit(unit, config, plan, seed, model, verified["cells"])
    if read(unit / "static_unit_audit.json") != current:
        raise ValueError("shared replay unit evidence changed before deserialization")
    recorded = config["device"]
    resolved = recorded if device == "auto" else device
    bundle, checkpoint = load_shared_neural_checkpoint(unit / "source_refit", device=resolved)
    return bundle, {"direction": plan["name"], "seed": seed, "model": model, "device": resolved,
        "original_training_device": recorded, "same_training_device": resolved == recorded,
        "unit_audit_sha256": stream_sha(unit / "static_unit_audit.json"), "model_source_paths": source_paths,
        "checkpoint": checkpoint, "training_performed": False}


def replay_unit(bundle, cells, plan, saved_arrays, *, origins=None, max_cells=None, progress_callback=None):
    """Every selected saved target point, plus exact paired causal controls.

    Saved FP32 match uses 1e-5 atol/rtol; same-device paired predictions require
    EXACT equality. An appended poisoned future row makes N100 non-vacuous even
    when the prepared representation itself ends at source cycle 100.
    """
    return replay_predictions(bundle, cells, plan, saved_arrays, predictor=predict_transfer_sequence, tolerance=1e-5,
        origins=origins, max_cells=max_cells, progress_callback=progress_callback)
