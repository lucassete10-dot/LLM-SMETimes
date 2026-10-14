"""Read-only shared-transfer metric/selection/weight audit; no model loading.

Static evidence does not prove executable predictive replay or deployment.
Source CV alone selects candidates/epochs; target errors are not a gate score.
"""

import copy
import json
from pathlib import Path

import numpy as np

from battery_smetimes.research_phase_gate import select_source_candidate
from battery_smetimes.research_inference import verify_source_snapshot
from battery_smetimes.research_transfer_baselines import _allowed_cells, transfer_input_schema, transfer_prefix
from battery_smetimes.research_transfer_signals import RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.audit_soh_research_artifacts_v5 import audit_scores, equal_number


STATUS = "passed shared source/target static numerical audit"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_coverage(report, ids, origins, cells):
    """Recompute eligibility, so a valid hard cell cannot be replaced by a skip."""
    predicted = {(r["cell_id"], r["origin"]) for r in report["by_cell"]}
    skipped = {(r["cell_id"], r["origin"]): r["reason"] for r in report["skipped"]}
    for key in ids:
        cell = cells[key]
        for origin in origins:
            pair, reason = (key, origin), None
            future = np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin)
            if future.sum() < 2:
                reason = "insufficient valid future targets"
            else:
                try:
                    transfer_prefix(cell, origin)
                except ValueError as error:
                    if "insufficient" in str(error) or "not established" in str(error):
                        reason = str(error)
                    else:
                        raise
            if (reason is None and pair not in predicted) or (reason is not None and skipped.get(pair) != reason):
                raise ValueError("shared coverage does not match original input/target eligibility; no error-based skips")


def audit_checkpoint(folder, source, plan, origins, seed, model, parameters, ids, cells, *, fold=None, epochs=None, device=None):
    folder, settings = Path(folder), source["configuration"]
    if (folder / "failure.json").exists():
        raise ValueError("unresolved failed shared checkpoint")
    expected = {"plan": plan, "origins": origins, "model": model, "parameters": parameters, "seed": seed,
                "configuration": settings, "source_fold": fold, "fixed_refit_epochs": epochs, "selection_uses_target_labels": False}
    config = read(folder / "config.json")
    if any(config[key] != value for key, value in expected.items()) or (device is not None and config["device"] != device):
        raise ValueError("shared source unit configuration/budget/roles changed")
    manifest = read(folder / "checkpoint_manifest.json")
    if (manifest["schema"] != 1 or not manifest["trusted_local_only"]
            or set(manifest["artifact_sha256"]) != {"model.pt", "model.json", "preprocessor.joblib"}):
        raise ValueError("shared checkpoint fingerprint membership incomplete")
    for name, digest in manifest["artifact_sha256"].items():
        if stream_sha(folder / name) != digest:
            raise ValueError("shared checkpoint artifact changed")
    info, resources = read(folder / "model.json"), read(folder / "resources.json")
    allowed = _allowed_cells(cells, plan)
    if (info["schema"] != "research_shared_transfer_sequence_v1" or info["mode"] != RESEARCH_MODE
            or not info["not_a_production_route"] or info["plan"] != plan or info["allowed_cells"] != allowed
            or info["input_schema"] != transfer_input_schema(cells[ids[0]]) or info["model_name"] != model
            or info["source_train_ids"] != ids or info["source_only_preprocessor_fit_ids"] != ids
            or info["origins"] != origins or info["parameters"] != parameters or info["seed"] != seed
            or info["device"] != config["device"] or info["history_width"] != 23 or info["statistics_width"] != 48
            or info["checkpoint_files_sha256"] != source["checkpoint_files_sha256"]):
        raise ValueError("shared model identity/source/scaler metadata differs from frozen contract")
    if info["llm_config"] != (settings if model.startswith("llm_") else None):
        raise ValueError("shared LLM settings changed")
    for name, value in (("source_train_ids", ids), ("source_only_preprocessor_fit_ids", ids),
                        ("source_validation_ids", fold["source_validation"] if fold else []), ("device", config["device"]),
                        ("total_parameters", info["total_parameters"]), ("trainable_parameters", info["trainable_parameters"]),
                        ("model_bytes", (folder / "model.pt").stat().st_size)):
        if resources[name] != value:
            raise ValueError("shared resource/scaler-fit identity differs from model")
    if not (0 < info["trainable_parameters"] <= info["total_parameters"]):
        raise ValueError("invalid shared model parameter accounting")
    history, report = read(folder / "training_history.json"), read(folder / "report.json")
    if (not report["complete"] or report["target_evaluated_during_training"] or not history
            or info["training_history"] != history or [r["epoch"] for r in history] != list(range(1, len(history)+1))
            or len(history) > settings["max_epochs"] or any(not np.isfinite(r["train_mse"]) or r["train_mse"] < 0 for r in history)):
        raise ValueError("shared source training history/target usage invalid")
    if fold:
        checks = [r for r in history if "source_validation_mape_percent_origin100" in r]
        if not checks or any(not np.isfinite(r["source_validation_mape_percent_origin100"]) for r in checks):
            raise ValueError("shared fold has no complete finite source validation")
        best = min(checks, key=lambda r: r["source_validation_mape_percent_origin100"])
        if report["selected_epochs"] != best["epoch"] or info["epochs"] != best["epoch"]:
            raise ValueError("shared selected epoch is not the source validation optimum")
        validation = audit_scores(folder / "source_validation", fold["source_validation"], origins,
                                  plan["source_physical_family"], cells)
        assert_coverage(validation, fold["source_validation"], origins, cells)
        best_report = audit_scores(folder / "best_source_validation", fold["source_validation"], origins, plan["source_physical_family"], cells)
        assert_coverage(best_report, fold["source_validation"], origins, cells)
        equal_number(validation["summary"]["100"]["cell_macro_mape_percent"], best["source_validation_mape_percent_origin100"], "source epoch optimum")
        equal_number(report["source_validation_mape_percent_origin100"], validation["summary"]["100"]["cell_macro_mape_percent"], "source fold report")
        if report["source_validation_by_cell"] != validation["by_cell"]:
            raise ValueError("source selection records differ from audited source predictions")
        primary = [r["cell_id"] for r in validation["by_cell"] if r["origin"] == 100]
        if len(primary) != len(fold["source_validation"]) or set(primary) != set(fold["source_validation"]):
            raise ValueError("primary source validation cells cannot disappear")
    else:
        if (len(history) != epochs or info["epochs"] != epochs or report["selected_epochs"] != epochs
                or report["source_validation_by_cell"] != [] or report["source_validation_mape_percent_origin100"] is not None
                or any("source_validation_mape_percent_origin100" in r for r in history)):
            raise ValueError("fixed source refit did not preserve source-selected epochs")
        validation = None
    return report, validation


def audit_shared_neural_unit(unit, run_config, plan, seed, model, cells):
    unit = Path(unit)
    source, origins = run_config["model_sources"][model], run_config["origins"]
    settings = source["configuration"]
    folds = plan["source_selection_folds"][:1] if run_config["smoke"] else plan["source_selection_folds"]
    if read(unit / "config.json") != {"plan": plan, "seed": seed, "model": model, "source": source, "folds": folds}:
        raise ValueError("shared transfer unit identity/configuration differs")
    expected_dirs = {f"candidate_{i}" for i in range(len(settings["candidates"]))}
    if {p.name for p in unit.glob("candidate_*") if p.is_dir()} != expected_dirs:
        raise ValueError("shared candidate budget differs from declared source search")
    candidates, epochs = [], {}
    for index, parameters in enumerate(settings["candidates"]):
        candidate = unit / f"candidate_{index}"
        if {p.name for p in candidate.iterdir() if p.is_dir()} != {p["name"] for p in folds}:
            raise ValueError("shared candidate source folds incomplete or extra")
        rows, epochs[index] = [], []
        for fold in folds:
            report, validation = audit_checkpoint(candidate / fold["name"], source, plan, origins, seed, model,
                parameters, fold["source_train"], cells, fold=fold, device=run_config.get("device"))
            rows.extend(validation["by_cell"])
            epochs[index].append(report["selected_epochs"])
        candidates.append({"index": index, "parameters": parameters, "by_cell": rows})
    if read(unit / "source_candidates.json") != candidates:
        raise ValueError("source candidate aggregate differs from source-only prediction arrays")
    required = [key for fold in folds for key in fold["source_validation"]]
    selected, scores = select_source_candidate(candidates, required)
    selected = {**selected, "fixed_refit_epochs": int(np.ceil(np.median(epochs[selected["index"]]))),
                "source_fold_selected_epochs": epochs[selected["index"]]}
    if read(unit / "selected.json") != selected or read(unit / "source_validation_scores.json") != scores:
        raise ValueError("source candidate/epoch selection differs from source-only optimum")
    audit_checkpoint(unit / "source_refit", source, plan, origins, seed, model, selected["parameters"],
                     plan["source_refit"], cells, epochs=selected["fixed_refit_epochs"], device=run_config.get("device"))
    target = audit_scores(unit / "target_test", plan["target_test"], origins, plan["target_physical_family"], cells)
    assert_coverage(target, plan["target_test"], origins, cells)
    if read(unit / "target_evaluation_contract.json") != {"target_calibration": [], "target_labels_used_for_selection": False,
                                                         "physical_families_preserved": True, "not_same_type_deployment": True}:
        raise ValueError("target evaluation/research contract changed")
    hashes = {}
    for path in unit.rglob("*"):
        if path.is_file() and path.name != "static_unit_audit.json":
            if not path.resolve().is_relative_to(unit.resolve()):
                raise ValueError("shared transfer artifact escaped unit")
            hashes[str(path.relative_to(unit))] = stream_sha(path)
    return {"schema": 1, "status": STATUS, "direction": plan["name"], "seed": seed, "model": model,
        "source_validation_fold_count": len(folds), "candidate_count": len(candidates),
        "source_validation_cells": len(required), "target_cells": len(plan["target_test"]),
        "target_summary": target["summary"], "artifact_sha256": hashes,
        "scope": "static metrics/grid/coverage/source-only selection/epoch/model-byte provenance; not executable weight replay"}


def assert_shared_neural_smoke(root, expected, plans, cells):
    """Consume every same-code eight-unit smoke, never an accuracy threshold."""
    if expected["models"] != ["gru", "statistics_mlp", "llm_e1", "llm_e2"] or len(plans) != 2:
        raise ValueError("shared full prerequisite needs all four models and both directions")
    for folder in sorted(Path(root).iterdir(), reverse=True):
        try:
            config = verify_source_snapshot(folder)
            progress = read(folder / "progress.json")
            if (not config["smoke"] or config["prepare_only"] or not progress["complete"] or not progress["model_training_started"]
                    or (folder / "failure.json").exists() or (folder / "launcher_failure.json").exists()):
                continue
            for name in ("prepared", "prepared_config_sha256", "prepared_manifest_sha256", "prepared_splits_sha256",
                         "main_run_bindings", "source_sha256", "origins", "device", "models", "input_schema", "feature_names", "packages"):
                if config[name] != expected[name]:
                    raise ValueError("shared smoke source/input/model contract differs")
            sources = copy.deepcopy(expected["model_sources"])
            for source in sources.values():
                settings = source["configuration"]
                settings["candidates"] = settings["candidates"][:1]
                settings["max_epochs"] = settings["smoke_epochs"]
            if (config["model_sources"] != sources or config["seeds"] != expected["seeds"][:1]
                    or config["target_calibration"] != [] or not config["not_same_type_deployment"]
                    or config["global_summary_writer"] is not False or read(folder / "splits.json")["directions"] != plans):
                raise ValueError("shared smoke budget/cohort differs")
            completed = [(r["direction"], r["seed"], r["model"]) for r in progress["completed"]]
            grid = {(plan["name"], config["seeds"][0], model) for plan in plans for model in config["models"]}
            if (len(completed) != len(set(completed)) or set(completed) != grid
                    or progress["expected_final_model_units"] != len(grid)):
                raise ValueError("shared smoke lacks complete model/direction coverage")
            audits = []
            for plan in plans:
                for model in config["models"]:
                    unit = folder / plan["name"] / str(config["seeds"][0]) / model
                    saved = read(unit / "static_unit_audit.json")
                    current = audit_shared_neural_unit(unit, config, plan, config["seeds"][0], model, cells)
                    if saved != current:
                        raise ValueError("shared smoke static evidence stale")
                    audits.append({"direction": plan["name"], "model": model, "static_audit_sha256": stream_sha(unit / "static_unit_audit.json")})
            if read(folder / "static_run_audit.json") != audit_shared_neural_run(folder, cells):
                raise ValueError("shared smoke complete-run numerical evidence stale")
            return {"run": str(folder.resolve()), "config_sha256": stream_sha(folder / "config.json"),
                "progress_sha256": stream_sha(folder / "progress.json"), "units": audits,
                "scope": "complete same-source engineering smoke with rechecked static numerical evidence; no target-score threshold"}
        except (KeyError, ValueError, OSError, TypeError):
            continue
    raise ValueError("completed audited same-source shared neural smoke required before full training")


def audit_shared_neural_run(run, cells, *, require_complete=True):
    """Independent whole-grid recheck, including every source candidate fold.

    Worker finalization can check a fully populated grid before the final
    complete flag. External audit requires that flag as well. Evidence binds
    actual membership rather than a progress SHA that would become stale at
    finalization; every consuming call rechecks full current membership.
    """
    run = Path(run)
    config = verify_source_snapshot(run)
    progress, plans = read(run / "progress.json"), read(run / "splits.json")["directions"]
    if (config["prepare_only"] or not progress["model_training_started"] or (run / "failure.json").exists()
            or (run / "launcher_failure.json").exists() or (require_complete and not progress["complete"])):
        raise ValueError("shared run is not a successful complete fitted experiment")
    prepared = Path(config["prepared"])
    for name, field in (("config.json", "prepared_config_sha256"), ("manifest.json", "prepared_manifest_sha256"),
                        ("splits.json", "prepared_splits_sha256")):
        if stream_sha(prepared / name) != config[field]:
            raise ValueError("shared run prepared input binding changed")
    inputs = verify_source_snapshot(prepared)
    from scripts.train_soh_research_transfer_neural_v5 import derive_model_sources
    sources = derive_model_sources(config["main_run_bindings"], inputs, smoke=config["smoke"])
    seeds = inputs["training_seeds"][:1] if config["smoke"] else inputs["training_seeds"]
    if (config["model_sources"] != sources or config["seeds"] != seeds or config["origins"] != inputs["origins_source_cycles"]
            or config["input_schema"] != inputs["input_schema"] or config["feature_names"] != inputs["feature_names"]
            or config["models"] != ["gru", "statistics_mlp", "llm_e1", "llm_e2"]
            or plans != read(prepared / "splits.json")["directions"] or len(plans) != 2
            or config["target_calibration"] != [] or config["global_summary_writer"] is not False
            or not config["not_same_type_deployment"]):
        raise ValueError("shared run source budget/seed/schema/cohort changed")
    grid = {(p["name"], seed, model) for p in plans for seed in config["seeds"] for model in config["models"]}
    completed = [(r["direction"], r["seed"], r["model"]) for r in progress["completed"]]
    if len(completed) != len(set(completed)) or set(completed) != grid or progress["expected_final_model_units"] != len(grid):
        raise ValueError("shared run does not cover its complete model/seed/direction grid")
    units = []
    for plan in plans:
        for seed in config["seeds"]:
            for model in config["models"]:
                unit = run / plan["name"] / str(seed) / model
                current = audit_shared_neural_unit(unit, config, plan, seed, model, cells)
                if read(unit / "static_unit_audit.json") != current:
                    raise ValueError("shared unit numerical evidence no longer matches artifacts")
                units.append({"direction": plan["name"], "seed": seed, "model": model,
                    "unit_audit_sha256": stream_sha(unit / "static_unit_audit.json"),
                    "target_summary": current["target_summary"]})
    return {"schema": 1, "status": "passed complete shared neural static run audit", "units": units,
        "completed_grid": [list(key) for key in sorted(grid)], "finalized_units": len(grid),
        "run_config_sha256": stream_sha(run / "config.json"), "splits_sha256": stream_sha(run / "splits.json"),
        "source_sha256": config["source_sha256"], "prepared_manifest_sha256": config["prepared_manifest_sha256"],
        "scope": "all saved source CV candidates/final refits/target arrays and current full-grid membership; not executable weight replay or an independent blind test"}
