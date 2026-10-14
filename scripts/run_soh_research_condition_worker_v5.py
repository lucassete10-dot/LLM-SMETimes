"""Resume one condition-only job with frozen model sources; never main CV writes."""

import argparse
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import sys
import traceback
import hashlib

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
WORKSPACE = Path.cwd().resolve()
sys.path.insert(0, str(WORKSPACE))


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--worker-token", required=True)
    parser.add_argument("--inspect-only", action="store_true", help="Verify data/source imports only; return before any model fitting")
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(WORKSPACE / "experiments/soh_research_v5/condition_runs") or run == WORKSPACE / "experiments/soh_research_v5/condition_runs":
        raise ValueError("not an explicit condition experiment job")
    if not (WORKSPACE / "battery_smetimes").is_dir():
        raise ValueError("condition worker must start in the recorded project workspace")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for name, digest in config["execution_source_sha256"].items():
        source = run / "execution_snapshot" / name
        if not source.resolve().is_relative_to((run / "execution_snapshot").resolve()) or fingerprint(source) != digest:
            raise ValueError("condition execution snapshot missing/changed")
    if fingerprint(Path(__file__)) != config["execution_source_sha256"]["scripts/run_soh_research_condition_worker_v5.py"]:
        raise ValueError("worker entrypoint differs from this job's frozen execution source")
    import battery_smetimes
    import scripts
    battery_smetimes.__path__ = [str(run / "execution_snapshot/battery_smetimes"), *battery_smetimes.__path__]
    scripts.__path__ = [str(run / "execution_snapshot/scripts"), *scripts.__path__]
    from scripts.prepare_same_type_forecast import save_json, sha
    from battery_smetimes.research_conditions_execution import load_condition_contract, condition_job_configuration
    from battery_smetimes.research_inference import verify_source_snapshot
    from battery_smetimes.research_phase_gate import assert_main_complete, assert_main_numerical_audits
    from scripts.audit_soh_research_artifacts_v5 import audit_unit
    import psutil
    config = verify_source_snapshot(run)
    for name, digest in config["execution_source_sha256"].items():
        source = run / "execution_snapshot" / name
        if not source.resolve().is_relative_to((run / "execution_snapshot").resolve()) or sha(source) != digest:
            raise ValueError("condition execution snapshot missing/changed")
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt = run / "attempts" / session
    attempt.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(attempt / "worker.log", encoding="utf-8"), logging.StreamHandler()])
    identity = {"pid": os.getpid(), "process_create_time": psutil.Process().create_time(),
        "worker_token": args.worker_token, "run": str(run), "session": session}
    save_json(attempt / "config.json", {**identity, "run_config_sha256": sha(run / "config.json"),
        "inspect_only": args.inspect_only,
        "policy": "gate before fitting; preserve finalized units and validation candidates; archive unfinished attempts; original-seed restart, not optimizer continuation"})
    state = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    try:
        pointer = run / "active_condition_worker.json"
        if pointer.is_file():
            previous = json.loads(pointer.read_text(encoding="utf-8"))
            try:
                process = psutil.Process(previous["pid"])
                if (process.pid != os.getpid() and process.create_time() == previous["process_create_time"]
                        and previous["worker_token"] in process.cmdline()):
                    raise ValueError("condition job already has a live worker")
            except psutil.NoSuchProcess:
                pass
        save_json(pointer, identity)
        plan_folder = run / "frozen_condition_plan"
        if (sha(plan_folder / "config.json") != config["condition_plan_config_sha256"]
                or sha(plan_folder / "splits.json") != config["condition_plan_splits_sha256"]):
            raise ValueError("frozen condition plan changed")
        condition, frozen, registered, registry, plans = load_condition_contract(plan_folder)
        binding = config["bound_main_run"]
        if binding not in condition["main_runs_with_frozen_model_search_and_source"] or config["main_bindings"] != condition["main_runs_with_frozen_model_search_and_source"]:
            raise ValueError("condition worker source binding differs from frozen plan")
        source = Path(binding["run"])
        original = json.loads((source / "config.json").read_text(encoding="utf-8"))
        expected, selected_plans = condition_job_configuration(original, condition, plans, smoke=config["smoke"])
        for key, value in expected.items():
            if config[key] != value:
                raise ValueError("condition model budget/source/seeds differ from bound main configuration")
        if json.loads((run / "splits.json").read_text(encoding="utf-8"))["plans"] != selected_plans:
            raise ValueError("worker partitions differ from the frozen condition contract")
        if sha(run / "frozen_audit/config.json") != config["frozen_audit_config_sha256"]:
            raise ValueError("condition job's original data audit changed")
        if config["frozen_audit_files_sha256"] != {str(p.relative_to(run / "frozen_audit")): sha(p)
                for p in (run / "frozen_audit").rglob("*") if p.is_file()}:
            raise ValueError("condition copied frozen-audit file membership/contents changed")
        if not args.inspect_only:
            gate = assert_main_complete(config["main_bindings"])
            numeric = assert_main_numerical_audits(config["main_bindings"],
                WORKSPACE / "experiments/soh_research_v5/artifact_audits", completion=gate)
            save_json(attempt / "main_execution_gates.json", {"complete": gate, "numerical": numeric})
            if not config["smoke"]:
                from scripts.train_soh_research_conditions_v5 import assert_condition_smoke
                batch_state = json.loads((Path(config["condition_batch"]) / "progress.json").read_text(encoding="utf-8"))
                save_json(attempt / "condition_smoke_gate.json", assert_condition_smoke(
                    WORKSPACE / "experiments/soh_research_v5/condition_batches", batch_state["jobs_prepared"]))
        else:
            save_json(attempt / "main_execution_gates.json", {"not_evaluated": True,
                "reason": "inspection cannot fit, select or evaluate a model; not a training gate pass"})
        for package, expected_version in config.get("packages", {}).items():
            if version(package) != expected_version:
                raise ValueError(f"condition model package version changed: {package}")
        for package, expected_version in config.get("optional_packages", {}).items():
            if version(package) != expected_version:
                raise ValueError(f"condition optional model package version changed: {package}")
        if config.get("sequence_config", {}).get("kind") == "frozen_llm":
            from battery_smetimes.research_transfer_vi120 import stream_sha
            checkpoint = Path(config["sequence_config"]["checkpoint"])
            for name, digest in config["checkpoint_files_sha256"].items():
                if stream_sha(checkpoint / name) != digest:
                    raise ValueError("condition frozen LLM checkpoint changed")
        # Dedicated process: clear cached current model modules, then resolve
        # every original model import from this job's copied main snapshot.
        import battery_smetimes
        import scripts
        for name in list(sys.modules):
            if name.startswith("battery_smetimes.") or name.startswith("scripts."):
                del sys.modules[name]
        battery_smetimes.__path__ = [str(run / "source_snapshot/battery_smetimes"), str(WORKSPACE / "battery_smetimes")]
        scripts.__path__ = [str(run / "source_snapshot/scripts"), str(WORKSPACE / "scripts")]
        from scripts.train_same_type_forecast import load_data
        from battery_smetimes.physical_types import require_same_registered_family
        prepared = Path(json.loads((run / "frozen_audit/audit.json").read_text(encoding="utf-8"))["prepared_folder"])
        manifest, cells = load_data(prepared, frozen["config"]["integrity_audit"])
        if config["prepared_files_sha256"] != {r["cell_id"]: r["sha256"] for r in manifest["cells"]}:
            raise ValueError("condition cells differ from the original main data fingerprints")
        for key, cell in cells.items():
            cell.update({field: registered[key][field] for field in ("physical_family", "reference_established_source_cycle")})
        neural = "sequence_config" in config
        driver_file = "train_soh_research_sequences_v5.py" if neural else "train_soh_research_v5.py"
        driver = import_file("condition_frozen_driver", run / "source_snapshot" / driver_file)
        recovery = import_file("condition_frozen_recovery", run / "execution_snapshot/scripts/resume_soh_research_v5.py")
        recovery.ROOT = WORKSPACE  # archive confinement is the real workspace, not execution_snapshot
        models = config["architectures"] if neural else list(config["model_candidates"])
        if args.inspect_only:
            import importlib
            required_modules = ["battery_smetimes.research_baselines", "battery_smetimes.physical_types", "battery_smetimes.forecast"]
            if neural:
                required_modules.append("battery_smetimes.research_sequences")
                if config["sequence_config"].get("kind") == "frozen_llm":
                    required_modules.append("battery_smetimes.research_llm")
            elif any(model not in ("ridge", "extra_trees") for model in models):
                required_modules.append("battery_smetimes.research_extended_baselines")
            module_paths = {}
            for name in required_modules:
                path = Path(importlib.import_module(name).__file__).resolve()
                if not path.is_relative_to((run / "source_snapshot").resolve()):
                    raise ValueError("model dependency was imported outside the frozen main source")
                module_paths[name] = str(path)
            save_json(attempt / "inspection.json", {"source_imports": module_paths,
                "audited_native_cells_loaded": len(cells), "condition_plans": [p["name"] for p in selected_plans],
                "models": models, "seeds": config["seeds"], "model_training_performed": False,
                "model_instantiated": False, "prediction_accuracy_verified": False})
            save_json(attempt / "status.json", {"status": "inspection_only_complete", "model_training_performed": False})
            print(json.dumps({"run": str(run), "inspection": str((attempt / "inspection.json").resolve()),
                "model_training_performed": False}, indent=2))
            return
        grid = {(p["name"], seed, name) for p in selected_plans for seed in config["seeds"] for name in models}
        completed = [(row["fold"], row["seed"], row["model"]) for row in state["completed"]]
        if len(completed) != len(set(completed)) or set(completed) - grid or state["expected_units"] != len(grid):
            raise ValueError("condition progress membership contradicts its selected frozen grid")
        audit_config = dict(config)
        if neural and config["smoke"]:
            audit_config["sequence_config"] = {**config["sequence_config"],
                "candidates": config["sequence_config"]["candidates"][:1]}
        for plan in selected_plans:
            family = plan["physical_family"]
            require_same_registered_family(family,
                [cells[k]["physical_family"] for k in plan["train"] + plan["validation"] + plan["test"]], registry)
            for seed in config["seeds"]:
                for model in models:
                    unit = run / plan["name"] / str(seed) / model
                    key = (plan["name"], seed, model)
                    if key in completed:
                        evidence = audit_unit(unit, audit_config, plan, seed, model, registered, cells, {}, registry)
                        old = json.loads((unit / "artifact_audit.json").read_text(encoding="utf-8"))
                        if old["artifact_sha256"] != evidence["artifact_sha256"]:
                            raise ValueError("completed condition artifacts changed; refusing refit/reuse")
                        continue
                    unit.mkdir(parents=True, exist_ok=True)
                    candidates = (config["sequence_config"]["candidates"][:1] if config["smoke"] else config["sequence_config"]["candidates"]) if neural else config["model_candidates"][model]
                    unit_config = {"plan": plan, "seed": seed, "model": model, "candidates": candidates,
                        "smoke": config["smoke"], "physical_family": family}
                    if (unit / "config.json").is_file() and json.loads((unit / "config.json").read_text(encoding="utf-8")) != unit_config:
                        raise ValueError("condition unit configuration changed")
                    save_json(unit / "config.json", unit_config)
                    state["model_training_started"] = True
                    save_json(run / "progress.json", state)
                    logging.info("CONDITION FIT %s %s seed=%s", plan["name"], model, seed)
                    execute = recovery.recover_neural if neural else recovery.recover_baseline
                    report = execute(unit, run, session, driver, config, cells, registry, family, plan, seed, model, candidates, save_json)
                    # Smoke has one neural candidate; adapt only the audit view
                    # of that declared smoke budget, never the saved main config.
                    evidence = audit_unit(unit, audit_config, plan, seed, model, registered, cells, {}, registry)
                    save_json(unit / "artifact_audit.json", evidence)
                    state["completed"].append({"fold": plan["name"], "seed": seed, "model": model})
                    for origin, summary in report["summary"].items():
                        state["summary"].append({"fold": plan["name"], "dataset": plan["dataset"],
                            "heldout_condition": plan["heldout_condition"], "physical_family": family,
                            "seed": seed, "model": model, "origin": int(origin), **summary})
                    save_json(run / "progress.json", state)
                    save_json(attempt / "progress.json", {"last": state["completed"][-1], "completed_units": len(state["completed"])})
        state["complete"] = True
        save_json(run / "progress.json", state)
        save_json(run / "summary.json", state)
        save_json(attempt / "status.json", {"status": "condition_job_complete", "completed_units": len(state["completed"])})
    except BaseException:
        save_json(attempt / "failure.json", {"traceback": traceback.format_exc(),
            "model_training_started": state["model_training_started"], "completed": state["completed"]})
        raise


if __name__ == "__main__":
    main()
