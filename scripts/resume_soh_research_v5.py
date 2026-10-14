"""Recover interrupted fixed-plan runs using their immutable model source.

Completed model units and validation candidates are never fitted again.
An unfinished candidate has no saved optimizer, so its attempt is archived and
restarted at the original seed, not falsely described as epoch continuation.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]


def archive_partial(path, run, session):
    """Recoverable, explicitly workspace/run-confined move, never deletion."""
    path, run = Path(path).resolve(), Path(run).resolve()
    target = run / "interrupted_attempts" / session / path.relative_to(run)
    if not run.is_relative_to(ROOT) or not path.is_relative_to(run) or not target.resolve().is_relative_to(run):
        raise ValueError("refusing a move outside the explicit experiment run")
    if path == run or target.exists():
        raise ValueError("invalid recovery archive target")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return str(target)


def scores_complete(folder, partition):
    import numpy as np
    scores = folder / partition
    if not (scores / "metrics.json").is_file() or not (scores / "predictions.npz").is_file():
        return False
    try:
        report = json.loads((scores / "metrics.json").read_text())
        if not report["by_cell"]:
            return False
        with np.load(scores / "predictions.npz") as arrays:
            for row in report["by_cell"]:
                prefix = f'{row["cell_id"]}__{row["origin"]}__'
                for suffix in ("cycle", "truth", "prediction"):
                    if len(arrays[prefix + suffix]) != row["future_targets"]:
                        return False
                    if not np.isfinite(arrays[prefix + suffix]).all():
                        return False
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        return False
    return True


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--worker-token", required=True)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--no-global-summary", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / "experiments/soh_research_v5/runs"):
        raise ValueError("resume target is not an explicit workspace research run")
    import psutil
    pointer = run / "active_resume_worker.json"
    if pointer.is_file():
        prior = json.loads(pointer.read_text())
        try:
            process = psutil.Process(prior["pid"])
            if process.pid != os.getpid() and prior["worker_token"] in process.cmdline():
                raise ValueError("this run already has a live recovery worker")
        except psutil.NoSuchProcess:
            pass
    import battery_smetimes
    import scripts
    battery_smetimes.__path__ = [str(run / "source_snapshot/battery_smetimes"), *battery_smetimes.__path__]
    scripts.__path__ = [str(run / "source_snapshot/scripts"), *scripts.__path__]
    from battery_smetimes.research_inference import verify_source_snapshot
    from battery_smetimes.physical_types import load_registry, require_same_registered_family
    from scripts.prepare_same_type_forecast import sha, save_json
    from scripts.train_same_type_forecast import load_data
    import numpy as np
    config = verify_source_snapshot(run)
    if "numpy" in config and config["numpy"] != np.__version__:
        raise ValueError("numpy version differs from the original experiment")
    for package, expected_version in config.get("packages", {}).items():
        if version(package) != expected_version:
            raise ValueError(f"package version changed: {package}")
    if config.get("sequence_config", {}).get("kind") == "frozen_llm":
        checkpoint = Path(config["sequence_config"]["checkpoint"])
        for filename, expected_hash in config["checkpoint_files_sha256"].items():
            if sha(checkpoint / filename) != expected_hash:
                raise ValueError("frozen LLM checkpoint changed")
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    recovery = run / "resumes" / session
    recovery.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, recovery / "resume_source.py")
    identity = {"pid": os.getpid(), "worker_token": args.worker_token, "session": session,
        "process_create_time": psutil.Process().create_time(), "run": str(run)}
    save_json(pointer, identity)
    save_json(recovery / "config.json", {**identity, "script_sha256": sha(Path(__file__)),
        "source_execution": "original source_snapshot model/extractor/scaler code; recovery driver only is new",
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "max_units": args.max_units,
        "policy": "preserve completed units; reuse validation-only candidates; archive incomplete stages and restart their original seed; never use test scores for selection"})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(recovery / "worker.log", encoding="utf-8"), logging.StreamHandler()])
    frozen = json.loads((run / "frozen_audit/config.json").read_text())
    audit = json.loads((run / "frozen_audit/audit.json").read_text())
    registered = json.loads((run / "frozen_audit/cell_registry.json").read_text())["cells"]
    registry_path = run / "frozen_audit/source_snapshot/battery_physical_registry.json"
    if sha(registry_path) != frozen["registry_sha256"] or sha(config["config"]["integrity_audit"]) != frozen["integrity_sha256"]:
        raise ValueError("recovery audit/quality rules changed")
    prepared = Path(audit["prepared_folder"])
    if sha(prepared / "manifest.json") != frozen["prepared_manifest_sha256"]:
        raise ValueError("recovery data manifest changed")
    registry = load_registry(registry_path)
    _, cells = load_data(prepared, config["config"]["integrity_audit"])
    for key, cell in cells.items():
        cell.update({name: registered[key][name] for name in ("physical_family", "reference_established_source_cycle")})
    plans = json.loads((run / "splits.json").read_text())["plans"]
    original_plans = json.loads((run / "frozen_audit/splits.json").read_text())["plans"]
    if any(plan not in original_plans for plan in plans):
        raise ValueError("recovery split changed")
    state = json.loads((run / "progress.json").read_text())
    neural = "sequence_config" in config
    model_names = config["architectures"] if neural else list(config["model_candidates"])
    driver_name = "train_soh_research_sequences_v5.py" if neural else "train_soh_research_v5.py"
    driver = import_file("frozen_driver", run / "source_snapshot" / driver_name)
    completed = {(r["fold"], r["seed"], r["model"]) for r in state["completed"]}
    units = [(plan, seed, name) for plan in plans for seed in config["seeds"] for name in model_names]
    recovered = 0
    try:
        for plan, seed, name in units:
            folder = run / plan["name"] / str(seed) / name
            key = (plan["name"], seed, name)
            if key in completed:
                if not scores_complete(folder, "test") or not (folder / "resources.json").is_file():
                    raise ValueError("a recorded completed unit is incomplete; requires artifact audit")
                continue
            family = registered[plan["train"][0]]["physical_family"]
            require_same_registered_family(family, [cells[k]["physical_family"] for k in plan["train"] + plan["validation"] + plan["test"]], registry)
            folder.mkdir(parents=True, exist_ok=True)
            candidates = (config["sequence_config"]["candidates"][:1] if config["smoke"] else config["sequence_config"]["candidates"]) if neural else config["model_candidates"][name]
            expected = {"plan": plan, "seed": seed, "model": name, "candidates": candidates,
                        "smoke": config["smoke"], "physical_family": family}
            if (folder / "config.json").exists():
                if json.loads((folder / "config.json").read_text()) != expected:
                    raise ValueError("recovery model configuration differs from the original")
            else:
                save_json(folder / "config.json", expected)
            logging.info("RESUME UNIT %s seed=%s %s", plan["name"], seed, name)
            if neural:
                report = recover_neural(folder, run, session, driver, config, cells, registry, family, plan, seed, name, candidates, save_json)
            else:
                report = recover_baseline(folder, run, session, driver, config, cells, registry, family, plan, seed, name, candidates, save_json)
            state["completed"].append({"fold": plan["name"], "seed": seed, "model": name})
            for origin, row in report["summary"].items():
                state["summary"].append({"fold": plan["name"], "seed": seed, "model": name, "origin": int(origin), **row})
            save_json(run / "progress.json", state)
            recovered += 1
            save_json(recovery / "progress.json", {"recovered_units": recovered, "last": state["completed"][-1]})
            if not args.no_global_summary:
                current_summary = import_file("recovery_summary", ROOT / "scripts/summarize_soh_research_v5.py")
                current_summary.summarize()
            logging.info("SAVED RECOVERED %s %s seed=%s", plan["name"], name, seed)
            if args.max_units and recovered >= args.max_units:
                break
        if len(state["completed"]) == len(units):
            state["complete"] = True
            save_json(run / "progress.json", state)
            save_json(run / "summary.json", state)
        save_json(recovery / "status.json", {"status": "run_complete" if state["complete"] else "bounded_recovery_complete",
            "recovered_units": recovered, "total_completed": len(state["completed"])})
    except BaseException:
        save_json(recovery / "failure.json", {"traceback": traceback.format_exc(), "recovered_units": recovered})
        raise


def recover_baseline(folder, run, session, driver, config, cells, registry, family, plan, seed, name, candidates, save_json):
    import joblib
    scores = []
    for index, parameters in enumerate(candidates):
        candidate = folder / f"candidate_{index}"
        if not (candidate / "model.joblib").is_file() or not scores_complete(candidate, "validation"):
            if candidate.exists():
                archive_partial(candidate, run, session)
            candidate.mkdir()
            bundle = driver.fit_baseline(cells, plan["train"], family, registry, config["config"],
                                       name, parameters, seed, config["smoke"])
            joblib.dump(bundle, candidate / "model.joblib")
            report, arrays = driver.evaluate_baseline(bundle, cells, plan["validation"])
            driver.save_scores(candidate / "validation", report, arrays)
        else:
            bundle = joblib.load(candidate / "model.joblib")
            report = json.loads((candidate / "validation/metrics.json").read_text())
        if bundle["parameters"] != parameters or bundle["train_ids"] != plan["train"] or bundle["seed"] != seed:
            raise ValueError("saved candidate does not match original validation fit")
        scores.append({"index": index, "parameters": parameters,
            "validation_mape_percent": report["summary"]["100"]["cell_macro_mape_percent"],
            "fit_seconds": bundle["fit_seconds"], "fit_warnings": bundle.get("fit_warnings", [])})
        save_json(folder / "validation_scores.json", scores)
    selected = min(scores, key=lambda r: r["validation_mape_percent"])
    if (folder / "selected.json").exists() and json.loads((folder / "selected.json").read_text())["parameters"] != selected["parameters"]:
        raise ValueError("validation selection changed during recovery")
    save_json(folder / "selected.json", selected)
    if (folder / "model.joblib").exists():
        bundle = joblib.load(folder / "model.joblib")
    else:
        bundle = driver.fit_baseline(cells, plan["train"] + plan["validation"], family, registry, config["config"],
                                   name, selected["parameters"], seed, config["smoke"])
        joblib.dump(bundle, folder / "model.joblib")
    if bundle["parameters"] != selected["parameters"] or bundle["train_ids"] != plan["train"] + plan["validation"]:
        raise ValueError("saved final fit contradicts original validation selection")
    started = time.perf_counter()
    if scores_complete(folder, "test"):
        report = json.loads((folder / "test/metrics.json").read_text())
    else:
        if (folder / "test").exists():
            archive_partial(folder / "test", run, session)
        report, arrays = driver.evaluate_baseline(bundle, cells, plan["test"])
        driver.save_scores(folder / "test", report, arrays)
    save_json(folder / "resources.json", {"fit_seconds": bundle["fit_seconds"],
        "recovery_test_save_seconds": time.perf_counter() - started,
        "saved_model_bytes": (folder / "model.joblib").stat().st_size,
        "training_pairs": bundle["training_pairs"], "gpu_used": False, "recovery_session": session})
    save_json(folder / "fit_diagnostics.json", {key: bundle.get(key) for key in (
        "fit_warnings", "iterations", "dual_gap", "fitted_kernel", "estimator_training_pairs", "coreset_by_cell")})
    return report


def recover_neural(folder, run, session, driver, config, cells, registry, family, plan, seed, name, candidates, save_json):
    import joblib
    import numpy as np
    import torch
    from battery_smetimes.research_sequences import HistoryPreprocessor, training_contexts, fit_epoch, predict_sequence
    settings = config["sequence_config"]
    llm = settings if settings.get("kind") == "frozen_llm" else None
    if config["torch"] != torch.__version__:
        raise ValueError("torch version differs from the original training")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = config["device"]
    predictor = predict_sequence
    if llm:
        from battery_smetimes.research_llm import attach_cards, fit_llm_epoch, adapter_state, load_adapter, predict_llm
        predictor = predict_llm
    def make_bundle(processor, network, ids):
        bundle = {"processor": processor, "network": network, "registry": registry, "physical_family": family,
            "device": device, "model_name": name, "train_ids": ids, "origins": config["config"]["origins_source_cycles"], "seed": seed}
        if llm:
            bundle.update(llm_config=llm, checkpoint_files_sha256=config["checkpoint_files_sha256"])
        return bundle
    def train_epoch(network, contexts, optimizer, epoch):
        if llm:
            return fit_llm_epoch(network, contexts, optimizer, seed, epoch, settings["batch_contexts"], device, llm["balance_coefficient"])
        return fit_epoch(network, contexts, optimizer, seed, epoch, settings["batch_contexts"], device)
    processor = HistoryPreprocessor().fit(cells, plan["train"], config["config"]["origins_source_cycles"])
    contexts = training_contexts(cells, plan["train"], processor, settings["max_future_targets_per_context"])
    scores = []
    max_epochs = settings["smoke_epochs"] if config["smoke"] else settings["max_epochs"]
    for index, parameters in enumerate(candidates):
        candidate = folder / f"candidate_{index}"
        ready = all((candidate / file).is_file() for file in ("model.pt", "model.json", "preprocessor.joblib")) and scores_complete(candidate, "validation")
        if ready:
            info = json.loads((candidate / "model.json").read_text())
            if info["parameters"] != parameters or info["seed"] != seed or info["train_ids"] != plan["train"]:
                raise ValueError("saved neural candidate differs from the frozen validation setting")
            report = json.loads((candidate / "validation/metrics.json").read_text())
            best = report["summary"]["100"]["cell_macro_mape_percent"]
            best_epoch = info["epochs"]
        else:
            if candidate.exists():
                archive_partial(candidate, run, session)
            candidate.mkdir()
            network = driver.instantiate(processor, name, parameters, seed, device, llm)
            bundle = make_bundle(processor, network, plan["train"])
            if llm:
                attach_cards(bundle, contexts, cells)
            optimizer = torch.optim.AdamW((p for p in network.parameters() if p.requires_grad),
                lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
            best, best_epoch, best_state, bad, history = float("inf"), 0, None, 0, []
            for epoch in range(1, max_epochs + 1):
                loss = train_epoch(network, contexts, optimizer, epoch)
                row = {"epoch": epoch, "train_mse": loss, **getattr(network, "last_epoch_diagnostics", {})}
                if epoch % settings["validation_every_epochs"] == 0 or epoch == max_epochs:
                    report, arrays = driver.evaluate_baseline(bundle, cells, plan["validation"], predictor=predictor)
                    value = report["summary"]["100"]["cell_macro_mape_percent"]
                    if value is None or not np.isfinite(value):
                        raise ValueError("nonfinite original validation criterion")
                    row["validation_cell_macro_mape_percent_origin100"] = value
                    if value < best:
                        best, best_epoch, bad = value, epoch, 0
                        best_state = adapter_state(network) if llm else {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
                    else:
                        bad += 1
                history.append(row)
                save_json(candidate / "training_history.json", history)
                logging.info("RECOVERY EPOCH %s %s seed=%s candidate=%s epoch=%s val=%s", plan["name"], name, seed, index, epoch,
                             row.get("validation_cell_macro_mape_percent_origin100"))
                if bad >= settings["patience_validations"]:
                    break
            if llm:
                load_adapter(network, best_state)
            else:
                network.load_state_dict(best_state)
            driver.save_network(candidate, bundle, parameters, best_epoch, history)
            report, arrays = driver.evaluate_baseline(bundle, cells, plan["validation"], predictor=predictor)
            driver.save_scores(candidate / "validation", report, arrays)
            del bundle, network, optimizer, best_state
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        scores.append({"index": index, "parameters": parameters, "validation_mape_percent": best, "selected_epochs": best_epoch})
        save_json(folder / "validation_scores.json", scores)
    selected = min(scores, key=lambda row: row["validation_mape_percent"])
    if (folder / "selected.json").exists():
        old = json.loads((folder / "selected.json").read_text())
        if old["parameters"] != selected["parameters"] or old["selected_epochs"] != selected["selected_epochs"]:
            raise ValueError("saved neural validation selection changed on recovery")
    save_json(folder / "selected.json", selected)
    ids = plan["train"] + plan["validation"]
    processor = HistoryPreprocessor().fit(cells, ids, config["config"]["origins_source_cycles"])
    network = driver.instantiate(processor, name, selected["parameters"], seed, device, llm)
    bundle = make_bundle(processor, network, ids)
    started = time.perf_counter()
    if all((folder / file).is_file() for file in ("model.pt", "model.json", "preprocessor.joblib")):
        info = json.loads((folder / "model.json").read_text())
        if info["parameters"] != selected["parameters"] or info["epochs"] != selected["selected_epochs"] or info["train_ids"] != ids:
            raise ValueError("saved final neural fit contradicts selected validation setting")
        bundle["processor"] = joblib.load(folder / "preprocessor.joblib")
        weights = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        if llm:
            load_adapter(network, weights)
        else:
            network.load_state_dict(weights)
    else:
        for file in ("model.pt", "model.json", "preprocessor.joblib", "refit_history.json"):
            if (folder / file).exists():
                archive_partial(folder / file, run, session)
        contexts = training_contexts(cells, ids, processor, settings["max_future_targets_per_context"])
        if llm:
            attach_cards(bundle, contexts, cells)
        optimizer = torch.optim.AdamW((p for p in network.parameters() if p.requires_grad),
            lr=selected["parameters"]["learning_rate"], weight_decay=selected["parameters"]["weight_decay"])
        history = []
        for epoch in range(1, selected["selected_epochs"] + 1):
            loss = train_epoch(network, contexts, optimizer, epoch)
            history.append({"epoch": epoch, "train_mse": loss, **getattr(network, "last_epoch_diagnostics", {})})
            save_json(folder / "refit_history.json", history)
        driver.save_network(folder, bundle, selected["parameters"], selected["selected_epochs"], history)
        del optimizer
    if scores_complete(folder, "test"):
        report = json.loads((folder / "test/metrics.json").read_text())
    else:
        if (folder / "test").exists():
            archive_partial(folder / "test", run, session)
        report, arrays = driver.evaluate_baseline(bundle, cells, plan["test"], predictor=predictor)
        driver.save_scores(folder / "test", report, arrays)
    save_json(folder / "resources.json", {"recovery_refit_and_test_seconds": time.perf_counter() - started,
        "recovery_session": session, "device": device, "training_cells": len(ids),
        "parameters": sum(p.numel() for p in network.parameters()),
        "cuda_peak_allocated_mb_cumulative_process": torch.cuda.max_memory_allocated() / 2 ** 20 if device.startswith("cuda") else None})
    del bundle, network, contexts
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    main()
