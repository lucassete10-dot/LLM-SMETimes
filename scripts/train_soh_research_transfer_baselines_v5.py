"""Frozen-source/recoverable source-CV shared baselines, separate from old runs."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "2")
WORKSPACE = Path.cwd().resolve()
sys.path.insert(0, str(WORKSPACE))
# No workspace module imports before the saved worker installs frozen paths.
SOURCES = ("scripts/train_soh_research_transfer_baselines_v5.py", "scripts/train_soh_research_transfer_v5.py",
    "scripts/replay_soh_research_shared_v5.py", "battery_smetimes/research_transfer_baseline_replay.py",
    "battery_smetimes/research_transfer_replay_checks.py",
    "battery_smetimes/research_transfer_baseline_execution.py", "battery_smetimes/research_transfer_baseline_audit.py",
    "battery_smetimes/research_transfer_baselines.py", "battery_smetimes/research_transfer_signals.py",
    "battery_smetimes/research_transfer_vi120.py", "battery_smetimes/research_phase_gate.py",
    "battery_smetimes/research_transfer_neural_audit.py", "battery_smetimes/research_transfer_neural_recovery.py",
    "battery_smetimes/research_inference.py", "scripts/audit_soh_research_artifacts_v5.py",
    "battery_smetimes/research_baselines.py", "battery_smetimes/research_generalization_splits.py",
    "battery_smetimes/research_splits.py", "battery_smetimes/physical_types.py", "battery_smetimes/forecast.py",
    "scripts/train_same_type_forecast.py", "scripts/prepare_same_type_forecast.py", "scripts/train_soh_research_v5.py",
    "scripts/summarize_soh_research_v5.py", "prepare_mit.py", "prepare_xjtu.py",
    "adapt_nasa_mat_v2.py", "audit_mit_label_free_extraction_v2.py", "battery_v2_features.py")
ORIGINAL_MODEL_FILES = ("battery_smetimes/research_baselines.py", "battery_smetimes/forecast.py", "battery_smetimes/physical_types.py")


def derive_baseline_source(bindings, inputs, *, smoke):
    from scripts.prepare_same_type_forecast import sha
    from battery_smetimes.research_inference import verify_source_snapshot
    matches = []
    for binding in bindings:
        if sha(Path(binding["run"]) / "config.json") != binding["config_sha256"]:
            raise ValueError("baseline original main configuration changed")
        main = verify_source_snapshot(Path(binding["run"]))
        if "architectures" in main:
            continue
        candidates = main["model_candidates"] if "model_candidates" in main else main["config"]["initial_candidates"]
        if set(candidates) != {"ridge", "extra_trees"}:
            continue
        if (main["smoke"] or main["seeds"] != inputs["training_seeds"]
                or main["config"]["origins_source_cycles"] != inputs["origins_source_cycles"]
                or main["tree_estimators"] != 160 or any(len(values) != 3 for values in candidates.values())):
            raise ValueError("baseline original budget/seed/origin differs from full main")
        matches.append({"binding": binding, "original_candidates": copy.deepcopy(candidates),
            "candidates": {model: copy.deepcopy(values[:1] if smoke else values) for model, values in candidates.items()},
            "tree_estimators": 48 if smoke else 160})
    if len(matches) != 1:
        raise ValueError("need exactly one unchanged original full Ridge/ExtraTrees budget source")
    return matches[0]


def prepare(args):
    from battery_smetimes.research_phase_gate import main_run_bindings
    from battery_smetimes.research_inference import verify_source_snapshot
    from scripts.prepare_same_type_forecast import save_json, sha
    pointer = "latest_vi_transfer_inputs.json" if args.protocol == "strict_vi120" else "latest_transfer_inputs.json"
    prepared = (args.prepared or Path(json.loads((WORKSPACE / "experiments/soh_research_v5" / pointer).read_text())["folder"])).resolve()
    inputs = verify_source_snapshot(prepared)
    if not json.loads((prepared / "progress.json").read_text())["prepared_complete"]:
        raise ValueError("shared baseline input preparation incomplete")
    bindings = main_run_bindings(WORKSPACE / "experiments/soh_research_v5/runs", inputs["audit_config_sha256"])
    source = derive_baseline_source(bindings, inputs, smoke=args.smoke)
    # Pin the native files themselves. The workspace's baseline module now also
    # dispatches newer baselines; do not replace the original Ridge/ET reference.
    original_snapshot = Path(source["binding"]["run"]) / "source_snapshot"
    original_hashes = {name: sha(original_snapshot / name) for name in ORIGINAL_MODEL_FILES}
    run = WORKSPACE / "experiments/soh_research_v5/transfer_baseline_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in SOURCES:
        path = run / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((original_snapshot if name in ORIGINAL_MODEL_FILES else WORKSPACE) / name, path)
        hashes[name] = sha(path)
    plans = json.loads((prepared / "splits.json").read_text())["directions"]
    seeds = inputs["training_seeds"][:1] if args.smoke else inputs["training_seeds"]
    config = {"prepared": str(prepared), "prepared_config_sha256": sha(prepared / "config.json"),
        "prepared_manifest_sha256": sha(prepared / "manifest.json"), "prepared_splits_sha256": sha(prepared / "splits.json"),
        "main_run_bindings": bindings, "baseline_source": source, "source_sha256": hashes, "original_model_sha256": original_hashes,
        "smoke": args.smoke, "prepare_only": args.prepare_only, "seeds": seeds, "origins": inputs["origins_source_cycles"],
        "device": "cpu", "models": ["ridge", "extra_trees"], "target_calibration": [], "not_same_type_deployment": True,
        "global_summary_writer": False, "input_schema": inputs["input_schema"], "feature_names": inputs["feature_names"],
        "packages": {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "scikit-learn", "joblib", "psutil", "h5py", "pandas")}}
    save_json(run / "config.json", config)
    save_json(run / "splits.json", {"directions": plans})
    save_json(run / "progress.json", {"complete": False, "model_training_started": False, "completed": [],
        "expected_final_model_units": len(plans)*len(seeds)*2})
    return run


def run_worker(run, *, resume):
    sys.path.insert(0, str(run / "source_snapshot"))
    import battery_smetimes
    import scripts
    battery_smetimes.__path__ = [str(run / "source_snapshot/battery_smetimes"), *battery_smetimes.__path__]
    scripts.__path__ = [str(run / "source_snapshot/scripts"), *scripts.__path__]
    from battery_smetimes.research_inference import verify_source_snapshot
    from battery_smetimes.research_transfer_neural_recovery import exclusive_run_lock, archive_interrupted, verify_root_utility_resolution
    from scripts.prepare_same_type_forecast import save_json, sha
    config = verify_source_snapshot(run)
    if sha(Path(__file__)) != config["source_sha256"]["scripts/train_soh_research_transfer_baselines_v5.py"]:
        raise ValueError("shared baseline worker entrypoint differs from its frozen source")
    with exclusive_run_lock(run):
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        attempt = run / "resumes" / session
        attempt.mkdir(parents=True, exist_ok=False)
        save_json(attempt / "config.json", {"pid": os.getpid(), "resume": resume, "run_config_sha256": sha(run / "config.json"),
            "source_sha256": config["source_sha256"], "recovery_policy": "reuse statically audited complete stages; original-seed restart for incomplete fits"})
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
            logging.FileHandler(attempt / "worker.log", encoding="utf-8"), logging.StreamHandler()])
        state = json.loads((run / "progress.json").read_text())
        try:
            if config["prepare_only"]:
                raise ValueError("baseline prepared-only contract cannot be repurposed into a fitted run")
            save_json(attempt / "root_utility_resolution.json", verify_root_utility_resolution(run, config))
            if not resume and (state["complete"] or state["completed"] or state["model_training_started"]
                    or (run / "failure.json").exists() or (run / "launcher_failure.json").exists()):
                raise ValueError("existing baseline training/failure requires explicit --resume")
            from battery_smetimes.research_phase_gate import assert_main_complete, assert_main_numerical_audits
            main = assert_main_complete(config["main_run_bindings"])
            numerical = assert_main_numerical_audits(config["main_run_bindings"], WORKSPACE / "experiments/soh_research_v5/artifact_audits", completion=main)
            save_json(attempt / "main_gates.json", {"completion": main, "numerical": numerical})
            from battery_smetimes.research_transfer_baseline_audit import verify_baseline_contract, audit_baseline_run, assert_baseline_smoke
            from battery_smetimes.research_transfer_baseline_execution import execute_baseline_unit, audit_baseline_unit
            from scripts.train_soh_research_transfer_v5 import load_shared_cells
            checked_config, plans = verify_baseline_contract(run)
            if checked_config != config:
                raise ValueError("baseline current verified configuration changed")
            cells = load_shared_cells(Path(config["prepared"]), progress_callback=lambda key, count, total:
                logging.info("BASELINE INPUT %s %s/%s", key, count, total))
            grid = {(p["name"], seed, model) for p in plans for seed in config["seeds"] for model in config["models"]}
            completed = [(r["direction"], r["seed"], r["model"]) for r in state["completed"]]
            if (len(completed) != len(set(completed)) or not set(completed).issubset(grid) or state["expected_final_model_units"] != len(grid)
                    or (state["complete"] and set(completed) != grid) or (completed and not state["model_training_started"])):
                raise ValueError("baseline recovery progress grid differs from frozen plan")
            for plan in plans:
                for seed in config["seeds"]:
                    for model in config["models"]:
                        if (plan["name"], seed, model) in completed:
                            unit = run / plan["name"] / str(seed) / model
                            if json.loads((unit / "static_unit_audit.json").read_text()) != audit_baseline_unit(unit, config, plan, seed, model, cells):
                                raise ValueError("baseline completed recovery unit changed")
            if not config["smoke"]:
                save_json(attempt / "verified_smoke_gate.json", assert_baseline_smoke(run.parent, config, cells))
            if resume:
                for name in ("failure.json", "launcher_failure.json"):
                    if (run / name).exists():
                        archive_interrupted(run, run / name, session)
            if state["complete"]:
                if json.loads((run / "static_run_audit.json").read_text()) != audit_baseline_run(run, cells):
                    raise ValueError("baseline completed run evidence changed")
                save_json(attempt / "result.json", {"complete": True, "no_fit": True})
                return
            for plan in plans:
                for seed in config["seeds"]:
                    for model in config["models"]:
                        if (plan["name"], seed, model) in completed:
                            continue
                        logging.info("BASELINE UNIT %s seed=%s model=%s", plan["name"], seed, model)
                        state["model_training_started"] = True
                        save_json(run / "progress.json", state)
                        execute_baseline_unit(run, config, plan, seed, model, cells, session=session, resume=resume)
                        state["completed"].append({"direction": plan["name"], "seed": seed, "model": model})
                        save_json(run / "progress.json", state)
            save_json(run / "static_run_audit.json", audit_baseline_run(run, cells, require_complete=False))
            state["complete"] = True
            save_json(run / "progress.json", state)
            save_json(attempt / "result.json", {"complete": True, "finalized_units": len(state["completed"])})
        except BaseException:
            failure = {"traceback": traceback.format_exc(), "state": state, "session": str(attempt)}
            save_json(attempt / "failure.json", failure)
            if not (run / "failure.json").exists():
                save_json(run / "failure.json", failure)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--protocol", choices=("strict_vi120", "cached_vit"), default="strict_vi120")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (WORKSPACE / "battery_smetimes").is_dir():
        raise ValueError("baseline execution must start in the original workspace")
    if args.run:
        run = args.run.resolve()
        root = WORKSPACE / "experiments/soh_research_v5/transfer_baseline_runs"
        if not run.is_relative_to(root) or run == root:
            raise ValueError("not an explicit shared baseline experiment path")
        run_worker(run, resume=args.resume)
    else:
        if args.resume:
            raise ValueError("--resume needs an explicit saved --run")
        run = prepare(args)
        print(json.dumps({"run": str(run), "prepare_only": args.prepare_only, "model_training_started": False}, indent=2), flush=True)
        if not args.prepare_only:
            command = [sys.executable, str(run / "source_snapshot/scripts/train_soh_research_transfer_baselines_v5.py"), "--run", str(run)]
            try:
                with (run / "worker.stdout.log").open("a", encoding="utf-8") as stdout, (run / "worker.stderr.log").open("a", encoding="utf-8") as stderr:
                    subprocess.run(command, cwd=WORKSPACE, check=True, stdout=stdout, stderr=stderr)
            except BaseException:
                from scripts.prepare_same_type_forecast import save_json
                save_json(run / "launcher_failure.json", {"traceback": traceback.format_exc(), "command": command,
                    "progress": json.loads((run / "progress.json").read_text()), "worker_stdout": str(run / "worker.stdout.log"),
                    "worker_stderr": str(run / "worker.stderr.log")})
                raise


if __name__ == "__main__":
    main()
