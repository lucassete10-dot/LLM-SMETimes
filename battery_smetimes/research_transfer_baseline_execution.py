"""Audited/recoverable Ridge and ExtraTrees shared-input research units.

No public launcher bypass here. The worker verifies original main completion,
data, versions, frozen source, source-only budgets and smoke before calling.
"""

import importlib.metadata
from pathlib import Path
import traceback

import joblib
import numpy as np
import psutil

from battery_smetimes.research_phase_gate import select_source_candidate
from battery_smetimes.research_transfer_baselines import (
    fit_transfer, evaluate_transfer, training_pairs, _allowed_cells, transfer_input_schema,
)
from battery_smetimes.research_transfer_neural_audit import assert_coverage, read
from battery_smetimes.research_transfer_neural_recovery import confined, archive_interrupted, journal
from battery_smetimes.research_transfer_signals import RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.audit_soh_research_artifacts_v5 import audit_scores
from scripts.prepare_same_type_forecast import save_json


INFO_FIELDS = ("schema", "mode", "input_schema", "plan", "allowed_cells", "source_train_ids", "used_source_train_ids",
    "source_only_preprocessor_fit_ids", "model_name", "parameters", "seed", "origins", "training_pairs", "smoke", "not_a_production_route")


def stage_config(plan, origins, model, parameters, seed, smoke, fold):
    if fold is not None:
        if (fold not in plan["source_selection_folds"] or not fold["source_validation"]
                or len(fold["source_validation"]) != len(set(fold["source_validation"]))
                or set(fold["source_train"]) & set(fold["source_validation"])
                or set(fold["source_train"] + fold["source_validation"]) != set(plan["source_refit"])):
            raise ValueError("baseline source fold roles changed/incomplete/overlapping")
        ids = fold["source_train"]
    else:
        ids = plan["source_refit"]
    if not ids or len(ids) != len(set(ids)) or set(ids)-set(plan["source_refit"]):
        raise ValueError("baseline source-only training IDs invalid")
    return {"plan": plan, "origins": origins, "model": model, "parameters": parameters, "seed": seed,
        "smoke": smoke, "source_fold": fold, "source_train_ids": ids, "selection_uses_target_labels": False, "device": "cpu"}


def checkpoint_info(bundle):
    info = {key: bundle[key] for key in INFO_FIELDS}
    info.update(numpy=importlib.metadata.version("numpy"), sklearn=importlib.metadata.version("scikit-learn"),
        joblib=importlib.metadata.version("joblib"), n_features=int(bundle["model"].n_features_in_),
        tree_estimators=len(bundle["model"].estimators_) if bundle["model_name"] == "extra_trees" else None,
        tree_nodes=sum(tree.tree_.node_count for tree in bundle["model"].estimators_) if bundle["model_name"] == "extra_trees" else None,
        linear_coefficients=int(bundle["model"].coef_.size) if bundle["model_name"] == "ridge" else None,
        fit_seconds=bundle["fit_seconds"], device="cpu")
    return info


def train_baseline_stage(cells, plan, origins, model, parameters, seed, smoke, destination, *, fold=None):
    contract = stage_config(plan, origins, model, parameters, seed, smoke, fold)
    _allowed_cells(cells, plan)  # Before creating any model or output directory.
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "config.json", contract)
    try:
        bundle = fit_transfer(cells, contract["source_train_ids"], plan, origins, model, parameters, seed, smoke)
        joblib.dump(bundle, destination / "model.joblib")
        info = checkpoint_info(bundle)
        save_json(destination / "model.json", info)
        save_json(destination / "resources.json", {"fit_seconds": bundle["fit_seconds"], "device": "cpu",
            "source_only_preprocessor_fit_ids": bundle["source_only_preprocessor_fit_ids"],
            "source_validation_ids": fold["source_validation"] if fold else [], "training_pairs": bundle["training_pairs"],
            "model_bytes": (destination / "model.joblib").stat().st_size, "n_features": info["n_features"],
            "tree_estimators": info["tree_estimators"], "tree_nodes": info["tree_nodes"], "linear_coefficients": info["linear_coefficients"],
            "process_rss_mb_at_end_not_peak": psutil.Process().memory_info().rss/2**20,
            "not_an_idle_inference_benchmark": True, "target_calibration": []})
        save_json(destination / "checkpoint_manifest.json", {"schema": 1, "trusted_local_only": True,
            "artifact_sha256": {name: stream_sha(destination / name) for name in ("model.joblib", "model.json", "resources.json")},
            "scope": "trusted-local byte fingerprint, not a signature or executable weight replay"})
        validation = None
        if fold:
            validation, arrays = evaluate_transfer(bundle, cells, fold["source_validation"])
            primary = [r["cell_id"] for r in validation["by_cell"] if r["origin"] == 100]
            if len(primary) != len(fold["source_validation"]) or set(primary) != set(fold["source_validation"]):
                raise ValueError("baseline source selection must retain every primary validation cell")
            from scripts.train_soh_research_v5 import save_scores
            save_scores(destination / "source_validation", validation, arrays)
        report = {"complete": True, "target_evaluated_during_training": False,
            "source_validation_by_cell": validation["by_cell"] if validation else []}
        save_json(destination / "report.json", report)
        return bundle, report
    except BaseException:
        save_json(destination / "failure.json", {"traceback": traceback.format_exc(), "target_evaluated_during_training": False})
        raise


def audit_baseline_stage(folder, cells, plan, origins, model, parameters, seed, smoke, *, fold=None):
    """Static source role/pair-count/validation/fingerprint check, no load/fit."""
    folder = Path(folder)
    expected = stage_config(plan, origins, model, parameters, seed, smoke, fold)
    if read(folder / "config.json") != expected or (folder / "failure.json").exists():
        raise ValueError("baseline source stage configuration/roles changed or unresolved failure")
    manifest = read(folder / "checkpoint_manifest.json")
    if (manifest["schema"] != 1 or not manifest["trusted_local_only"]
            or set(manifest["artifact_sha256"]) != {"model.joblib", "model.json", "resources.json"}):
        raise ValueError("baseline checkpoint fingerprint membership invalid")
    for name, digest in manifest["artifact_sha256"].items():
        if stream_sha(folder / name) != digest:
            raise ValueError("baseline checkpoint artifact changed")
    info, resources = read(folder / "model.json"), read(folder / "resources.json")
    x, y, _, used = training_pairs(cells, expected["source_train_ids"], origins)
    identity = {"schema": "research_shared_transfer_baseline_v1", "mode": RESEARCH_MODE, "not_a_production_route": True,
        "plan": plan, "allowed_cells": _allowed_cells(cells, plan), "input_schema": transfer_input_schema(cells[expected["source_train_ids"][0]]),
        "source_train_ids": expected["source_train_ids"], "used_source_train_ids": used,
        "source_only_preprocessor_fit_ids": expected["source_train_ids"], "model_name": model, "parameters": parameters,
        "seed": seed, "origins": origins, "smoke": smoke, "training_pairs": len(y), "n_features": x.shape[1], "device": "cpu",
        "tree_estimators": (48 if smoke else 160) if model == "extra_trees" else None,
        "linear_coefficients": x.shape[1] if model == "ridge" else None}
    if any(info[key] != value for key, value in identity.items()):
        raise ValueError("baseline checkpoint source/scaler/schema/estimator identity invalid")
    for key, package in (("numpy", "numpy"), ("sklearn", "scikit-learn"), ("joblib", "joblib")):
        if info[key] != importlib.metadata.version(package):
            raise ValueError("baseline checkpoint package version differs")
    if not np.isfinite(info["fit_seconds"]) or info["fit_seconds"] < 0:
        raise ValueError("invalid baseline fitting resource")
    for name, value in (("source_only_preprocessor_fit_ids", expected["source_train_ids"]),
                        ("source_validation_ids", fold["source_validation"] if fold else []), ("device", "cpu"),
                        ("training_pairs", len(y)), ("n_features", x.shape[1]), ("fit_seconds", info["fit_seconds"]),
                        ("model_bytes", (folder / "model.joblib").stat().st_size), ("target_calibration", []),
                        ("tree_estimators", info["tree_estimators"]), ("tree_nodes", info["tree_nodes"]),
                        ("linear_coefficients", info["linear_coefficients"])):
        if resources[name] != value:
            raise ValueError("baseline fitting resource/role accounting changed")
    report = read(folder / "report.json")
    if not report["complete"] or report["target_evaluated_during_training"]:
        raise ValueError("baseline stage not complete or evaluated target during fitting")
    if fold:
        validation = audit_scores(folder / "source_validation", fold["source_validation"], origins, plan["source_physical_family"], cells)
        assert_coverage(validation, fold["source_validation"], origins, cells)
        primary = [r["cell_id"] for r in validation["by_cell"] if r["origin"] == 100]
        if len(primary) != len(fold["source_validation"]) or set(primary) != set(fold["source_validation"]):
            raise ValueError("baseline primary validation cell missing")
        if report["source_validation_by_cell"] != validation["by_cell"]:
            raise ValueError("baseline source selection records differ from actual validation arrays")
    elif report["source_validation_by_cell"] != []:
        raise ValueError("all-source baseline refit must not reselect on validation or target")
    return report


def load_baseline_checkpoint(folder):
    """Trusted local only; caller checked parent and static source stage first."""
    folder = Path(folder)
    manifest = read(folder / "checkpoint_manifest.json")
    if (manifest["schema"] != 1 or not manifest["trusted_local_only"]
            or set(manifest["artifact_sha256"]) != {"model.joblib", "model.json", "resources.json"}):
        raise ValueError("baseline checkpoint manifest invalid")
    for name, digest in manifest["artifact_sha256"].items():
        if stream_sha(folder / name) != digest:
            raise ValueError("baseline weights/metadata/resource bytes changed before loading")
    info = read(folder / "model.json")
    for key, package in (("numpy", "numpy"), ("sklearn", "scikit-learn"), ("joblib", "joblib")):
        if info[key] != importlib.metadata.version(package):
            raise ValueError("baseline checkpoint runtime changed")
    bundle = joblib.load(folder / "model.joblib")
    if checkpoint_info(bundle) != info:
        raise ValueError("baseline deserialized model identity differs from saved metadata")
    estimator = bundle["model"]
    if estimator.n_features_in_ != 51 or bundle["imputer"].n_features_in_ != 51:
        raise ValueError("baseline saved feature dimensions invalid")
    if bundle["mode"] != RESEARCH_MODE or not bundle["not_a_production_route"]:
        raise ValueError("baseline saved model is not an explicit research-only route")
    if bundle["source_only_preprocessor_fit_ids"] != bundle["source_train_ids"]:
        raise ValueError("baseline saved scaler fit IDs invalid")
    if bundle["model_name"] == "ridge":
        if estimator.alpha != bundle["parameters"]["alpha"] or estimator.solver != "svd" or bundle["scaler"].n_features_in_ != 51:
            raise ValueError("baseline saved ridge/scaler settings invalid")
    elif bundle["model_name"] == "extra_trees":
        if (estimator.n_estimators != (48 if bundle["smoke"] else 160) or estimator.random_state != bundle["seed"]
                or estimator.min_samples_leaf != bundle["parameters"]["min_samples_leaf"] or estimator.max_features != .8
                or estimator.n_jobs != 4 or bundle["scaler"] is not None):
            raise ValueError("baseline saved forest settings invalid")
    else:
        raise ValueError("unsupported baseline checkpoint")
    return bundle


def recover_baseline_stage(cells, plan, origins, model, parameters, seed, smoke, destination,
                           *, run, session, resume, fold=None, need_bundle=False):
    _, destination = confined(run, destination)
    expected = stage_config(plan, origins, model, parameters, seed, smoke, fold)
    if destination.exists():
        if not resume:
            raise ValueError("existing baseline stage requires explicit resume")
        if not destination.is_dir():
            raise ValueError("baseline source stage is not a directory")
        if any(destination.iterdir()):
            if not (destination / "config.json").is_file() or read(destination / "config.json") != expected:
                raise ValueError("interrupted baseline configuration/seed/roles changed or missing")
            if (destination / "report.json").exists() and read(destination / "report.json").get("complete"):
                report = audit_baseline_stage(destination, cells, plan, origins, model, parameters, seed, smoke, fold=fold)
                journal(run, session, {"event": "reuse_audited_baseline_stage", "stage": str(destination.relative_to(Path(run).resolve())),
                    "seed": seed, "no_fit": True, "checkpoint_manifest_sha256": stream_sha(destination / "checkpoint_manifest.json")})
                return (load_baseline_checkpoint(destination) if need_bundle else None), report
        archive_interrupted(run, destination, session)
    bundle, report = train_baseline_stage(cells, plan, origins, model, parameters, seed, smoke, destination, fold=fold)
    journal(run, session, {"event": "baseline_stage_trained", "stage": str(destination.relative_to(Path(run).resolve())), "seed": seed})
    return (bundle if need_bundle else None), report


def unit_config(config, plan, seed, model):
    folds = plan["source_selection_folds"][:1] if config["smoke"] else plan["source_selection_folds"]
    return {"plan": plan, "seed": seed, "model": model, "origins": config["origins"], "smoke": config["smoke"],
        "baseline_source": config["baseline_source"], "folds": folds, "target_calibration": []}


TARGET_CONTRACT = {"target_calibration": [], "target_labels_used_for_selection": False,
    "physical_families_preserved": True, "not_same_type_deployment": True}


def audit_baseline_unit(unit, config, plan, seed, model, cells):
    unit = Path(unit)
    expected = unit_config(config, plan, seed, model)
    if read(unit / "config.json") != expected:
        raise ValueError("baseline unit source/budget/roles changed")
    settings, folds = config["baseline_source"]["candidates"][model], expected["folds"]
    if {p.name for p in unit.glob("candidate_*") if p.is_dir()} != {f"candidate_{i}" for i in range(len(settings))}:
        raise ValueError("baseline candidate budget differs from declared source search")
    candidates = []
    for index, parameters in enumerate(settings):
        candidate = unit / f"candidate_{index}"
        if {p.name for p in candidate.iterdir() if p.is_dir()} != {fold["name"] for fold in folds}:
            raise ValueError("baseline source folds incomplete or extra")
        rows = []
        for fold in folds:
            report = audit_baseline_stage(candidate / fold["name"], cells, plan, config["origins"], model, parameters,
                seed, config["smoke"], fold=fold)
            rows.extend(report["source_validation_by_cell"])
        candidates.append({"index": index, "parameters": parameters, "by_cell": rows})
    selected, scores = select_source_candidate(candidates, [key for fold in folds for key in fold["source_validation"]])
    if (read(unit / "source_candidates.json") != candidates or read(unit / "selected.json") != selected
            or read(unit / "source_validation_scores.json") != scores):
        raise ValueError("baseline source-only candidate selection changed")
    audit_baseline_stage(unit / "source_refit", cells, plan, config["origins"], model, selected["parameters"], seed, config["smoke"])
    target = audit_scores(unit / "target_test", plan["target_test"], config["origins"], plan["target_physical_family"], cells)
    assert_coverage(target, plan["target_test"], config["origins"], cells)
    if read(unit / "target_evaluation_contract.json") != TARGET_CONTRACT:
        raise ValueError("baseline target research contract changed")
    hashes = {}
    for path in unit.rglob("*"):
        if path.is_file() and path.name != "static_unit_audit.json":
            confined(unit, path)
            hashes[str(path.relative_to(unit))] = stream_sha(path)
    return {"schema": 1, "status": "passed shared baseline source/target static numerical audit", "direction": plan["name"],
        "seed": seed, "model": model, "source_fold_count": len(folds), "candidate_count": len(settings),
        "target_summary": target["summary"], "artifact_sha256": hashes, "scope": "static full-unit evidence; not executable weight replay"}


def execute_baseline_unit(run, config, plan, seed, model, cells, *, session, resume):
    _, unit = confined(run, Path(run) / plan["name"] / str(seed) / model)
    expected = unit_config(config, plan, seed, model)
    if unit.exists():
        if not resume:
            raise ValueError("existing baseline unit requires explicit resume")
        if any(unit.iterdir()) and (not (unit / "config.json").exists() or read(unit / "config.json") != expected):
            raise ValueError("baseline unit configuration missing or changed")
    unit.mkdir(parents=True, exist_ok=True)
    if (unit / "static_unit_audit.json").exists():
        current = audit_baseline_unit(unit, config, plan, seed, model, cells)
        if read(unit / "static_unit_audit.json") != current:
            raise ValueError("completed baseline unit changed; do not silently refit")
        journal(run, session, {"event": "reuse_complete_baseline_unit", "direction": plan["name"], "seed": seed, "model": model})
        return current
    if not (unit / "config.json").exists():
        save_json(unit / "config.json", expected)
    candidates = []
    for index, parameters in enumerate(config["baseline_source"]["candidates"][model]):
        rows = []
        for fold in expected["folds"]:
            _, report = recover_baseline_stage(cells, plan, config["origins"], model, parameters, seed, config["smoke"],
                unit / f"candidate_{index}" / fold["name"], run=run, session=session, resume=resume, fold=fold)
            rows.extend(report["source_validation_by_cell"])
        candidates.append({"index": index, "parameters": parameters, "by_cell": rows})
    path = unit / "source_candidates.json"
    if path.exists() and read(path) != candidates[:len(read(path))]:
        raise ValueError("saved baseline source candidates changed")
    save_json(path, candidates)
    selected, scores = select_source_candidate(candidates, [key for fold in expected["folds"] for key in fold["source_validation"]])
    for name, value in (("selected.json", selected), ("source_validation_scores.json", scores)):
        if (unit / name).exists() and read(unit / name) != value:
            raise ValueError("saved baseline source-only selection changed")
        if not (unit / name).exists():
            save_json(unit / name, value)
    evaluated = (unit / "target_evaluation_contract.json").exists()
    if evaluated and (read(unit / "target_evaluation_contract.json") != TARGET_CONTRACT
            or not (unit / "source_refit/report.json").exists() or not read(unit / "source_refit/report.json").get("complete")):
        raise ValueError("baseline target contract/refit completion changed")
    bundle, _ = recover_baseline_stage(cells, plan, config["origins"], model, selected["parameters"], seed, config["smoke"],
        unit / "source_refit", run=run, session=session, resume=resume, need_bundle=not evaluated)
    target_folder = unit / "target_test"
    if not evaluated:
        if target_folder.exists():
            archive_interrupted(run, target_folder, session)
        report, arrays = evaluate_transfer(bundle, cells, plan["target_test"])
        from scripts.train_soh_research_v5 import save_scores
        save_scores(target_folder, report, arrays)
        save_json(unit / "target_evaluation_contract.json", TARGET_CONTRACT)
    current = audit_baseline_unit(unit, config, plan, seed, model, cells)
    save_json(unit / "static_unit_audit.json", current)
    return current
