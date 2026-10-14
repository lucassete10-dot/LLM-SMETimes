"""Recover shared transfer stages without changing selection or restarting peers.

Only a caller holding the OS run lock may mutate a fitted run. Finished stages
are statically audited before reuse; interrupted stages restart at their original
seed, not from an optimizer/epoch checkpoint. Journals stay OUTSIDE model units.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil


def verify_root_utility_resolution(run, config):
    """Actual root-level extraction helpers, not just their saved copies."""
    import importlib
    from battery_smetimes.research_transfer_vi120 import stream_sha
    snapshot = Path(run).resolve() / "source_snapshot"
    paths = {}
    for name in ("prepare_mit", "prepare_xjtu", "adapt_nasa_mat_v2", "audit_mit_label_free_extraction_v2", "battery_v2_features"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(snapshot) or config["source_sha256"].get(name + ".py") != stream_sha(path):
            raise ValueError("raw utility implementation resolved outside verified frozen source: " + name)
        paths[name] = str(path)
    return paths


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def confined(run, path):
    run, path = Path(run).absolute(), Path(path).absolute()
    if not run.is_dir() or run == Path(run.anchor) or not path.is_relative_to(run) or path == run:
        raise ValueError("recovery target is not a bounded artifact inside the explicit run")
    for candidate in (run, *(run / part for part in path.relative_to(run).parents)):
        if candidate.is_symlink() or candidate.is_junction():
            raise ValueError("recovery paths must not traverse links or junctions")
    if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(run.resolve()):
        raise ValueError("recovery target escaped its explicit run")
    return run.resolve(), path.resolve()


@contextmanager
def exclusive_run_lock(run):
    """Nonblocking OS byte/file lock, automatically released on process death.

    The persisted PID is informational, NEVER evidence that a process died.
    Windows msvcrt and POSIX flock prevent two live workers from resuming even
    when progress/failure files are stale. No lock-file deletion is required.
    """
    run, path = confined(run, Path(run) / "worker.lock")
    handle = path.open("a+b")
    locked = False
    try:
        if path.stat().st_size == 0:
            handle.write(b"\n")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("shared neural run already has a live OS-lock owner; do not restart") from error
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def journal(run, session, event):
    from scripts.prepare_same_type_forecast import save_json
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError("invalid recovery session name")
    _, folder = confined(run, Path(run) / "resumes" / session)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "events.json"
    events = read(path) if path.exists() else []
    save_json(path, [*events, event])


def archive_interrupted(run, path, session):
    """Move an exact permitted stage to recoverable history, never delete it."""
    from battery_smetimes.research_transfer_vi120 import stream_sha
    run, path = confined(run, path)
    relative = path.relative_to(run)
    parts = relative.parts
    permitted = (len(parts) == 1 and parts[0] in {"failure.json", "launcher_failure.json"}) or (
        len(parts) == 4 and parts[3] in {"source_refit", "target_test"}) or (
        len(parts) == 5 and re.fullmatch(r"candidate_[0-9]+", parts[3]) is not None)
    if not permitted or not path.exists() or not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError("not an explicit permitted interrupted training/evaluation stage")
    _, destination = confined(run, run / "interrupted_attempts" / session / relative)
    if destination.exists():
        raise ValueError("recovery destination already exists; history must not be overwritten")
    contents = [path] if path.is_file() else list(path.rglob("*"))
    for entry in contents:
        confined(run, entry)
    files = [p for p in contents if p.is_file()]
    hashes = {}
    for file in files:
        confined(run, file)
        hashes[str(file.relative_to(run))] = stream_sha(file)
    journal(run, session, {"event": "archive_planned", "original": str(relative),
        "destination": str(destination.relative_to(run)), "artifact_sha256": hashes,
        "restart_policy": "original seed from start; no partial estimator or optimizer continuation"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    journal(run, session, {"event": "archive_completed", "original": str(relative),
        "destination": str(destination.relative_to(run))})
    return destination


def recover_source_stage(cells, plan, origins, model, parameters, seed, device, source, destination,
                         *, run, session, resume, fold=None, epochs=None, need_bundle=False):
    """Audit/reuse a completed stage, or archive/restart one incomplete stage.

    Caller verified frozen source/data/packages and holds exclusive_run_lock.
    A claimed completed but invalid stage is an ERROR, not an excuse to retrain.
    """
    from battery_smetimes.research_transfer_neural_audit import audit_checkpoint
    from battery_smetimes.research_transfer_neural_training import (
        train_shared_source_unit, load_shared_neural_checkpoint,
    )
    from battery_smetimes.research_transfer_vi120 import stream_sha
    _, destination = confined(run, destination)
    expected = {"plan": plan, "origins": origins, "model": model, "parameters": parameters,
        "seed": seed, "device": device, "configuration": source["configuration"], "source_fold": fold,
        "fixed_refit_epochs": epochs, "selection_uses_target_labels": False}
    if destination.exists():
        if not resume:
            raise ValueError("existing shared source stage requires explicit resume")
        if not destination.is_dir():
            raise ValueError("shared source stage is not a directory")
        if any(destination.iterdir()):
            if not (destination / "config.json").is_file() or read(destination / "config.json") != expected:
                raise ValueError("interrupted source configuration/seed/device/roles changed or missing")
            if (destination / "report.json").exists() and read(destination / "report.json").get("complete"):
                ids = fold["source_train"] if fold is not None else plan["source_refit"]
                report, _ = audit_checkpoint(destination, source, plan, origins, seed, model, parameters,
                    ids, cells, fold=fold, epochs=epochs)
                journal(run, session, {"event": "reuse_audited_source_stage",
                    "stage": str(destination.relative_to(Path(run).resolve())), "seed": seed,
                    "checkpoint_manifest_sha256": stream_sha(destination / "checkpoint_manifest.json"), "no_fit": True})
                bundle = load_shared_neural_checkpoint(destination, device=device)[0] if need_bundle else None
                return bundle, report
        archive_interrupted(run, destination, session)
    bundle, report = train_shared_source_unit(cells, plan, origins, model, parameters, seed, device,
        source["configuration"], destination, source_fold=fold, fixed_refit_epochs=epochs,
        llm_config=source["configuration"] if model.startswith("llm_") else None,
        checkpoint_hashes=source["checkpoint_files_sha256"])
    if not need_bundle:
        del bundle
        bundle = None
    journal(run, session, {"event": "source_stage_trained", "stage": str(destination.relative_to(Path(run).resolve())),
        "seed": seed, "selected_epochs": report["selected_epochs"], "optimizer_resumed": False})
    return bundle, report


def execute_shared_unit(run, config, plan, seed, model, cells, *, session, resume):
    """Resume whole source CV/refit/target unit, without using target selection."""
    import numpy as np
    import torch
    from battery_smetimes.research_phase_gate import select_source_candidate
    from battery_smetimes.research_transfer_neural_audit import audit_shared_neural_unit, assert_coverage
    from scripts.audit_soh_research_artifacts_v5 import audit_scores
    from battery_smetimes.research_transfer_baselines import evaluate_transfer
    from battery_smetimes.research_transfer_sequences import predict_transfer_sequence
    from scripts.prepare_same_type_forecast import save_json
    from scripts.train_soh_research_v5 import save_scores
    _, unit = confined(run, Path(run) / plan["name"] / str(seed) / model)
    source = config["model_sources"][model]
    settings = source["configuration"]
    folds = plan["source_selection_folds"][:1] if config["smoke"] else plan["source_selection_folds"]
    expected = {"plan": plan, "seed": seed, "model": model, "source": source, "folds": folds}
    if unit.exists():
        if not resume:
            raise ValueError("existing shared unit requires explicit resume")
        if any(unit.iterdir()) and (not (unit / "config.json").is_file() or read(unit / "config.json") != expected):
            raise ValueError("shared unit configuration changed or missing")
    unit.mkdir(parents=True, exist_ok=True)
    if (unit / "static_unit_audit.json").exists():
        current = audit_shared_neural_unit(unit, config, plan, seed, model, cells)
        if read(unit / "static_unit_audit.json") != current:
            raise ValueError("completed shared unit evidence changed; do not silently retrain")
        journal(run, session, {"event": "reuse_complete_unit", "direction": plan["name"], "seed": seed, "model": model})
        return current
    if not (unit / "config.json").exists():
        save_json(unit / "config.json", expected)
    candidates, epochs = [], {}
    for index, parameters in enumerate(settings["candidates"]):
        rows, epochs[index] = [], []
        for fold in folds:
            _, report = recover_source_stage(cells, plan, config["origins"], model, parameters, seed, config["device"],
                source, unit / f"candidate_{index}" / fold["name"], run=run, session=session,
                resume=resume, fold=fold)
            rows.extend(report["source_validation_by_cell"])
            epochs[index].append(report["selected_epochs"])
            if config["device"].startswith("cuda"):
                torch.cuda.empty_cache()
        candidates.append({"index": index, "parameters": parameters, "by_cell": rows})
    candidate_path = unit / "source_candidates.json"
    if candidate_path.exists() and read(candidate_path) != candidates[:len(read(candidate_path))]:
        raise ValueError("saved source candidate aggregate changed")
    save_json(candidate_path, candidates)
    selected, scores = select_source_candidate(candidates, [key for fold in folds for key in fold["source_validation"]])
    selected = {**selected, "fixed_refit_epochs": int(np.ceil(np.median(epochs[selected["index"]]))),
        "source_fold_selected_epochs": epochs[selected["index"]]}
    for name, value in (("selected.json", selected), ("source_validation_scores.json", scores)):
        if (unit / name).exists() and read(unit / name) != value:
            raise ValueError("saved source-only candidate/epoch selection changed")
        if not (unit / name).exists():
            save_json(unit / name, value)
    target_dir = unit / "target_test"
    contract = {"target_calibration": [], "target_labels_used_for_selection": False,
        "physical_families_preserved": True, "not_same_type_deployment": True}
    evaluated = (unit / "target_evaluation_contract.json").exists()
    if evaluated and read(unit / "target_evaluation_contract.json") != contract:
        raise ValueError("target evaluation contract changed")
    if evaluated and (not (unit / "source_refit/report.json").is_file()
            or not read(unit / "source_refit/report.json").get("complete")):
        raise ValueError("target evaluation cannot precede a completed source refit")
    bundle, _ = recover_source_stage(cells, plan, config["origins"], model, selected["parameters"], seed,
        config["device"], source, unit / "source_refit", run=run, session=session, resume=resume,
        epochs=selected["fixed_refit_epochs"], need_bundle=not evaluated)
    if evaluated:
        target = audit_scores(target_dir, plan["target_test"], config["origins"], plan["target_physical_family"], cells)
        assert_coverage(target, plan["target_test"], config["origins"], cells)
    else:
        if target_dir.exists():
            archive_interrupted(run, target_dir, session)
        target, arrays = evaluate_transfer(bundle, cells, plan["target_test"], predictor=predict_transfer_sequence)
        save_scores(target_dir, target, arrays)
        save_json(unit / "target_evaluation_contract.json", contract)
    del bundle
    if config["device"].startswith("cuda"):
        torch.cuda.empty_cache()
    audit = audit_shared_neural_unit(unit, config, plan, seed, model, cells)
    save_json(unit / "static_unit_audit.json", audit)
    return audit
