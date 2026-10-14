"""Frozen-source shared GRU/MLP/E1/E2 transfer, after completed main audits.

Prepare-only never instantiates or fits. The saved worker runs separately from
native models; source CV alone selects parameters and a median refit epoch.
No target calibration or main summary.csv writes. Explicit recovery reuses
audited stages; partial stages restart at the original seed. Cross-run
aggregation and public-data predictive replay remain separate work.
"""

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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
WORKSPACE = Path.cwd().resolve()
sys.path.insert(0, str(WORKSPACE))
SOURCES = (
    "scripts/train_soh_research_transfer_neural_v5.py", "scripts/train_soh_research_transfer_v5.py",
    "battery_smetimes/research_transfer_neural_training.py", "battery_smetimes/research_transfer_sequences.py",
    "battery_smetimes/research_transfer_neural_audit.py", "scripts/audit_soh_research_artifacts_v5.py",
    "battery_smetimes/research_transfer_neural_recovery.py",
    "battery_smetimes/research_transfer_neural_replay.py", "scripts/replay_soh_research_shared_v5.py",
    "battery_smetimes/research_transfer_replay_checks.py",
    "battery_smetimes/research_transfer_baselines.py", "battery_smetimes/research_transfer_signals.py",
    "battery_smetimes/research_transfer_vi120.py", "battery_smetimes/research_phase_gate.py",
    "battery_smetimes/research_inference.py", "battery_smetimes/research_conditions_execution.py",
    "battery_smetimes/research_sequences.py", "battery_smetimes/research_llm.py",
    "battery_smetimes/research_baselines.py", "battery_smetimes/research_extended_baselines.py",
    "battery_smetimes/research_generalization_splits.py", "battery_smetimes/physical_types.py",
    "battery_smetimes/forecast.py", "scripts/train_same_type_forecast.py",
    "scripts/prepare_same_type_forecast.py", "scripts/train_soh_research_v5.py", "prepare_mit.py",
    "prepare_xjtu.py", "adapt_nasa_mat_v2.py", "audit_mit_label_free_extraction_v2.py", "battery_v2_features.py",
    "battery_smetimes/research_splits.py", "scripts/summarize_soh_research_v5.py",
)


def derive_model_sources(bindings, inputs, *, smoke):
    from battery_smetimes.research_inference import verify_source_snapshot
    from scripts.prepare_same_type_forecast import sha
    model_sources = {}
    for binding in bindings:
        if sha(Path(binding["run"]) / "config.json") != binding["config_sha256"]:
            raise ValueError("shared model bound main configuration changed")
        main = verify_source_snapshot(Path(binding["run"]))
        if main["smoke"]:
            raise ValueError("shared model sources must be full native main configurations")
        for architecture in ("gru", "statistics_mlp", "llm_e1", "llm_e2"):
            if architecture in main.get("architectures", []):
                if architecture in model_sources:
                    raise ValueError("ambiguous main model source")
                if main["seeds"] != inputs["training_seeds"] or main["config"]["origins_source_cycles"] != inputs["origins_source_cycles"]:
                    raise ValueError("shared neural source budget differs from frozen main")
                config = copy.deepcopy(main["sequence_config"])
                if len(config["candidates"]) != 3:
                    raise ValueError("shared neural transfer requires the original three-candidate search")
                if smoke:
                    config["candidates"] = config["candidates"][:1]
                    config["max_epochs"] = config["smoke_epochs"]
                model_sources[architecture] = {"binding": binding, "configuration": config,
                    "checkpoint_files_sha256": main.get("checkpoint_files_sha256", {})}
    if set(model_sources) != {"gru", "statistics_mlp", "llm_e1", "llm_e2"}:
        raise ValueError("shared neural transfer needs all four bound original model settings")
    return model_sources


def prepare(args):
    from battery_smetimes.research_phase_gate import main_run_bindings
    from battery_smetimes.research_inference import verify_source_snapshot
    from scripts.prepare_same_type_forecast import save_json, sha
    pointer = "latest_vi_transfer_inputs.json" if args.protocol == "strict_vi120" else "latest_transfer_inputs.json"
    prepared = (args.prepared or Path(json.loads((WORKSPACE / "experiments/soh_research_v5" / pointer).read_text())["folder"])).resolve()
    inputs = verify_source_snapshot(prepared)
    if not json.loads((prepared / "progress.json").read_text())["prepared_complete"]:
        raise ValueError("shared transfer inputs incomplete")
    bindings = main_run_bindings(WORKSPACE / "experiments/soh_research_v5/runs", inputs["audit_config_sha256"])
    model_sources = derive_model_sources(bindings, inputs, smoke=args.smoke)
    # The architecture/optimizer/epoch implementations must still be exactly
    # the original main versions; only shared input/representation is new.
    for architecture, source in model_sources.items():
        for name in ("battery_smetimes/research_sequences.py", "battery_smetimes/research_baselines.py",
                     "battery_smetimes/forecast.py", "battery_smetimes/physical_types.py") + (
                        ("battery_smetimes/research_llm.py",) if architecture.startswith("llm_") else ()):
            if sha(WORKSPACE / name) != sha(Path(source["binding"]["run"]) / "source_snapshot" / name):
                raise ValueError("shared model algorithm changed from bound main source")
    run = WORKSPACE / "experiments/soh_research_v5/transfer_neural_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in SOURCES:
        destination = run / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE / name, destination)
        hashes[name] = sha(destination)
    plans = json.loads((prepared / "splits.json").read_text())["directions"]
    seeds = inputs["training_seeds"][:1] if args.smoke else inputs["training_seeds"]
    config = {"prepared": str(prepared), "prepared_config_sha256": sha(prepared / "config.json"),
        "prepared_manifest_sha256": sha(prepared / "manifest.json"), "prepared_splits_sha256": sha(prepared / "splits.json"),
        "main_run_bindings": bindings, "model_sources": model_sources, "source_sha256": hashes,
        "smoke": args.smoke, "prepare_only": args.prepare_only, "seeds": seeds,
        "origins": inputs["origins_source_cycles"], "device": args.device,
        "models": ["gru", "statistics_mlp", "llm_e1", "llm_e2"],
        "source_cv": "first fold only for engineering smoke; all source folds for full",
        "refit_epoch_rule": "ceil(median(selected epochs across chosen source CV folds)); never target errors",
        "comparison_baselines": "separate shared Ridge/ExtraTrees executor using identical prepared inputs and source-CV roles",
        "not_same_type_deployment": True, "target_calibration": [], "global_summary_writer": False,
        "input_schema": inputs["input_schema"], "feature_names": inputs["feature_names"],
        "packages": {key: importlib.metadata.version(key) for key in ("torch", "numpy", "scipy", "scikit-learn", "transformers", "joblib", "psutil", "h5py", "pandas")}}
    save_json(run / "config.json", config)
    save_json(run / "splits.json", {"directions": plans})
    save_json(run / "progress.json", {"complete": False, "model_training_started": False, "completed": [],
        "expected_final_model_units": len(plans)*len(seeds)*4})
    return run


def run_worker(run, *, resume=False):
    # No model imports before frozen path installation.
    sys.path.insert(0, str(run / "source_snapshot"))
    import battery_smetimes
    import scripts
    battery_smetimes.__path__ = [str(run / "source_snapshot/battery_smetimes"), *battery_smetimes.__path__]
    scripts.__path__ = [str(run / "source_snapshot/scripts"), *scripts.__path__]
    from scripts.prepare_same_type_forecast import save_json, sha
    from battery_smetimes.research_inference import verify_source_snapshot
    config = verify_source_snapshot(run)
    if sha(Path(__file__)) != config["source_sha256"]["scripts/train_soh_research_transfer_neural_v5.py"]:
        raise ValueError("shared neural worker entrypoint differs from saved source")
    from battery_smetimes.research_transfer_neural_recovery import exclusive_run_lock
    with exclusive_run_lock(run):
        run_locked_worker(run, config, resume=resume)


def run_locked_worker(run, config, *, resume):
    from scripts.prepare_same_type_forecast import save_json, sha
    from battery_smetimes.research_inference import verify_source_snapshot
    from battery_smetimes.research_transfer_neural_recovery import archive_interrupted, execute_shared_unit, verify_root_utility_resolution
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt = run / "resumes" / session
    attempt.mkdir(parents=True, exist_ok=False)
    save_json(attempt / "config.json", {"run_config_sha256": sha(run / "config.json"), "resume": resume,
        "pid": os.getpid(), "source_sha256": config["source_sha256"],
        "restart_policy": "reuse audited complete stages; original-seed epoch-1 restart for incomplete stages"})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(attempt / "worker.log", encoding="utf-8"), logging.StreamHandler()])
    state = json.loads((run / "progress.json").read_text())
    try:
        if config["prepare_only"]:
            raise ValueError("prepared-only contract cannot be repurposed into a fitted run")
        save_json(attempt / "root_utility_resolution.json", verify_root_utility_resolution(run, config))
        if not resume and (state["completed"] or state["model_training_started"] or state["complete"]
                or (run / "failure.json").exists() or (run / "launcher_failure.json").exists()):
            raise ValueError("existing shared training/failure requires explicit --resume; do not overwrite")
        from battery_smetimes.research_phase_gate import assert_main_complete, assert_main_numerical_audits, select_source_candidate
        main = assert_main_complete(config["main_run_bindings"])
        numerical = assert_main_numerical_audits(config["main_run_bindings"], WORKSPACE / "experiments/soh_research_v5/artifact_audits", completion=main)
        save_json(attempt / "main_gates.json", {"completion": main, "numerical": numerical})
        prepared = Path(config["prepared"])
        for name, field in (("config.json", "prepared_config_sha256"), ("manifest.json", "prepared_manifest_sha256"),
                            ("splits.json", "prepared_splits_sha256")):
            if sha(prepared / name) != config[field]:
                raise ValueError("shared neural prepared inputs changed")
        inputs = verify_source_snapshot(prepared)
        expected_models = derive_model_sources(config["main_run_bindings"], inputs, smoke=config["smoke"])
        expected_seeds = inputs["training_seeds"][:1] if config["smoke"] else inputs["training_seeds"]
        if (config["model_sources"] != expected_models or config["seeds"] != expected_seeds
                or config["origins"] != inputs["origins_source_cycles"]
                or config["input_schema"] != inputs["input_schema"] or config["feature_names"] != inputs["feature_names"]
                or config["models"] != ["gru", "statistics_mlp", "llm_e1", "llm_e2"]
                or config["target_calibration"] != [] or not config["not_same_type_deployment"]
                or config["global_summary_writer"] is not False):
            raise ValueError("shared neural budget/seed/schema/research-only contract changed")
        for name, expected in config["packages"].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError("shared neural package version changed")
        import torch
        from scripts.train_soh_research_transfer_v5 import load_shared_cells
        from battery_smetimes.research_transfer_neural_audit import audit_shared_neural_unit, assert_shared_neural_smoke, audit_shared_neural_run
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        cells = load_shared_cells(prepared)
        plans = json.loads((run / "splits.json").read_text())["directions"]
        if plans != json.loads((prepared / "splits.json").read_text())["directions"]:
            raise ValueError("shared neural roles changed from immutable prepared split")
        grid = {(plan["name"], seed, model) for plan in plans for seed in config["seeds"] for model in config["models"]}
        completed = [(row["direction"], row["seed"], row["model"]) for row in state["completed"]]
        if (len(completed) != len(set(completed)) or not set(completed).issubset(grid)
                or state["expected_final_model_units"] != len(grid)
                or (state["complete"] and set(completed) != grid)
                or (completed and not state["model_training_started"])):
            raise ValueError("shared recovery progress grid differs from frozen plan")
        for plan in plans:
            for seed in config["seeds"]:
                for model in config["models"]:
                    if (plan["name"], seed, model) in completed:
                        unit = run / plan["name"] / str(seed) / model
                        if json.loads((unit / "static_unit_audit.json").read_text()) != audit_shared_neural_unit(unit, config, plan, seed, model, cells):
                            raise ValueError("completed shared recovery unit changed; do not refit")
        if not config["smoke"]:
            save_json(run / "verified_shared_smoke_gate.json", assert_shared_neural_smoke(
                WORKSPACE / "experiments/soh_research_v5/transfer_neural_runs", config, plans, cells))
        # Only after frozen source/data/package/grid/gates and completed units
        # pass may old terminal failure records move to recoverable history.
        if resume:
            for name in ("failure.json", "launcher_failure.json"):
                if (run / name).exists():
                    archive_interrupted(run, run / name, session)
        if state["complete"]:
            if json.loads((run / "static_run_audit.json").read_text()) != audit_shared_neural_run(run, cells):
                raise ValueError("completed shared run audit changed")
            save_json(attempt / "result.json", {"complete": True, "no_fit": True, "reverified_finalized_units": len(grid)})
            return
        for plan in plans:
            for seed in config["seeds"]:
                for model in config["models"]:
                    if (plan["name"], seed, model) in completed:
                        continue
                    state["model_training_started"] = True
                    save_json(run / "progress.json", state)
                    execute_shared_unit(run, config, plan, seed, model, cells, session=session, resume=resume)
                    state["completed"].append({"direction": plan["name"], "seed": seed, "model": model})
                    save_json(run / "progress.json", state)
        save_json(run / "static_run_audit.json", audit_shared_neural_run(run, cells, require_complete=False))
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", type=Path, help="Internal saved worker entrypoint; no automatic restart")
    parser.add_argument("--resume", action="store_true", help="Explicit frozen-source recovery; never optimizer continuation")
    args = parser.parse_args()
    if not (WORKSPACE / "battery_smetimes").is_dir():
        raise ValueError("shared neural execution must start in the original workspace")
    if args.run:
        run = args.run.resolve()
        root = WORKSPACE / "experiments/soh_research_v5/transfer_neural_runs"
        if not run.is_relative_to(root) or run == root:
            raise ValueError("not an explicit shared neural experiment path")
        run_worker(run, resume=args.resume)
    else:
        if args.resume:
            raise ValueError("--resume requires an explicit saved --run")
        run = prepare(args)
        print(json.dumps({"run": str(run), "prepare_only": args.prepare_only, "model_training_started": False}, indent=2), flush=True)
        if not args.prepare_only:
            command = [sys.executable, str(run / "source_snapshot/scripts/train_soh_research_transfer_neural_v5.py"),
                       "--run", str(run)]
            try:
                with (run / "worker.stdout.log").open("a", encoding="utf-8") as stdout, (run / "worker.stderr.log").open("a", encoding="utf-8") as stderr:
                    subprocess.run(command, cwd=WORKSPACE, check=True, stdout=stdout, stderr=stderr)
            except BaseException:
                from scripts.prepare_same_type_forecast import save_json
                save_json(run / "launcher_failure.json", {"traceback": traceback.format_exc(), "command": command,
                    "progress": json.loads((run / "progress.json").read_text()),
                    "worker_stdout": str(run / "worker.stdout.log"), "worker_stderr": str(run / "worker.stderr.log")})
                raise


if __name__ == "__main__":
    main()
