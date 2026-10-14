"""Trusted-local frozen Ridge/ExtraTrees replay; never fit or select models."""

import inspect
from pathlib import Path

from battery_smetimes.research_inference import verify_source_snapshot
from battery_smetimes.research_phase_gate import assert_main_complete, assert_main_numerical_audits
from battery_smetimes.research_transfer_baseline_audit import verify_baseline_contract, audit_baseline_run
from battery_smetimes.research_transfer_baseline_execution import (
    audit_baseline_unit, load_baseline_checkpoint, checkpoint_info,
)
from battery_smetimes.research_transfer_baselines import predict_transfer, transfer_prefix, fit_transfer
from battery_smetimes.research_transfer_neural_audit import read
from battery_smetimes.research_transfer_neural_recovery import verify_root_utility_resolution
from battery_smetimes.research_transfer_replay_checks import prefix_only, replay_predictions, causal_predictions
from battery_smetimes.research_transfer_vi120 import stream_sha, SCHEMA as STRICT_RAW_SCHEMA
from scripts.train_soh_research_transfer_v5 import load_shared_cells


def verify_replay_run(run, *, progress_callback=None):
    run = Path(run).resolve()
    config = verify_source_snapshot(run)
    state = read(run / "progress.json")
    if (config["prepare_only"] or not state["complete"] or not state["model_training_started"]
            or (run / "failure.json").exists() or (run / "launcher_failure.json").exists()):
        raise ValueError("baseline replay requires successful completed real fitting, not prepared/partial/failed runs")
    config, plans = verify_baseline_contract(run)
    completion = assert_main_complete(config["main_run_bindings"])
    numerical = assert_main_numerical_audits(config["main_run_bindings"], run.parents[1] / "artifact_audits", completion=completion)
    root_paths = verify_root_utility_resolution(run, config)
    cells = load_shared_cells(Path(config["prepared"]), progress_callback=progress_callback)
    static = audit_baseline_run(run, cells)
    if read(run / "static_run_audit.json") != static:
        raise ValueError("baseline replay complete-run static evidence changed")
    return {"run": run, "config": config, "plans": plans, "cells": cells,
        "evidence": {"run": str(run), "run_config_sha256": stream_sha(run / "config.json"),
            "progress_sha256": stream_sha(run / "progress.json"), "splits_sha256": stream_sha(run / "splits.json"),
            "static_run_audit_sha256": stream_sha(run / "static_run_audit.json"),
            "prepared_manifest_sha256": config["prepared_manifest_sha256"], "source_sha256": config["source_sha256"],
            "original_model_sha256": config["original_model_sha256"],
            "main_completion": completion, "main_numerical": numerical, "static_finalized_units": static["finalized_units"],
            "root_utility_paths": root_paths, "shared_inputs_rederived": True,
            "strict_raw_signals_and_fingerprints_rechecked": config.get("input_schema") == STRICT_RAW_SCHEMA,
            "input_rederivation_scope": "raw MAT/HDF5 strict VI signals" if config.get("input_schema") == STRICT_RAW_SCHEMA else "shared signals from audited native prepared features; not a fresh raw MAT extraction",
            "weights_loaded": False, "training_performed": False, "not_same_type_deployment": True, "target_calibration": []}}


def verify_model_resolution(run, config):
    snapshot = (Path(run) / "source_snapshot").resolve()
    objects = {"checkpoint_loader": load_baseline_checkpoint, "checkpoint_identity": checkpoint_info,
        "predictor": predict_transfer, "prefix_features": transfer_prefix, "source_fit_implementation": fit_transfer,
        "replay_checks": replay_predictions, "prefix_removal": prefix_only, "causal_control": causal_predictions}
    paths = {}
    for name, value in objects.items():
        path = Path(inspect.getfile(value)).resolve()
        if not path.is_relative_to(snapshot):
            raise ValueError("baseline implementation resolved outside frozen source; use a fresh saved replay interpreter")
        relative = path.relative_to(snapshot).as_posix()
        if config["source_sha256"].get(relative) != stream_sha(path):
            raise ValueError("baseline predictor/check source is not in the verified snapshot")
        paths[name] = str(path)
    return paths


def load_verified_unit(verified, plan, seed, model, *, device="auto"):
    run, config, evidence = verified["run"], verified["config"], verified["evidence"]
    if plan not in verified["plans"] or seed not in config["seeds"] or model not in config["models"]:
        raise ValueError("baseline replay unit outside frozen direction/seed/model grid")
    if config["device"] != "cpu" or device not in ("auto", "cpu"):
        raise ValueError("baseline replay supports original CPU backend only")
    for name, key in (("config.json", "run_config_sha256"), ("progress.json", "progress_sha256"),
                      ("splits.json", "splits_sha256"), ("static_run_audit.json", "static_run_audit_sha256")):
        if stream_sha(run / name) != evidence[key]:
            raise ValueError("verified baseline parent evidence changed before deserialization")
    verify_source_snapshot(run)
    source_paths = verify_model_resolution(run, config)
    unit = run / plan["name"] / str(seed) / model
    current = audit_baseline_unit(unit, config, plan, seed, model, verified["cells"])
    if read(unit / "static_unit_audit.json") != current:
        raise ValueError("baseline replay unit evidence changed before deserialization")
    folder = unit / "source_refit"
    bundle = load_baseline_checkpoint(folder)
    return bundle, {"direction": plan["name"], "seed": seed, "model": model, "device": "cpu",
        "original_training_device": "cpu", "same_training_device": True,
        "unit_audit_sha256": stream_sha(unit / "static_unit_audit.json"), "model_source_paths": source_paths,
        "checkpoint": {"manifest_sha256": stream_sha(folder / "checkpoint_manifest.json"), "no_fit": True,
            "third_party_estimator_runtime": "original package versions, not vendored sklearn source"},
        "training_performed": False}


def replay_unit(bundle, cells, plan, saved_arrays, *, origins=None, max_cells=None, progress_callback=None):
    return replay_predictions(bundle, cells, plan, saved_arrays, predictor=predict_transfer, tolerance=1e-10,
        origins=origins, max_cells=max_cells, progress_callback=progress_callback)
