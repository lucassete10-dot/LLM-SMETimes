"""Prepare isolated frozen-source condition jobs; execute only after main audits."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from battery_smetimes.research_conditions_execution import load_condition_contract, condition_job_configuration
from battery_smetimes.research_phase_gate import assert_main_complete, assert_main_numerical_audits
from scripts.prepare_same_type_forecast import save_json, sha
from battery_smetimes.research_transfer_vi120 import stream_sha


WORKSPACE = Path(__file__).resolve().parents[1]
EXECUTION_SOURCES = (
    "scripts/train_soh_research_conditions_v5.py", "scripts/run_soh_research_condition_worker_v5.py",
    "scripts/resume_soh_research_v5.py", "scripts/audit_soh_research_artifacts_v5.py",
    "scripts/replay_soh_research_v5.py",
    "battery_smetimes/research_conditions_execution.py", "battery_smetimes/research_generalization_splits.py",
    "battery_smetimes/research_inference.py", "battery_smetimes/research_phase_gate.py",
    "battery_smetimes/research_transfer_vi120.py",
    "battery_smetimes/research_baselines.py", "battery_smetimes/physical_types.py",
    "battery_smetimes/research_splits.py", "battery_smetimes/forecast.py",
    "scripts/train_same_type_forecast.py", "scripts/prepare_same_type_forecast.py",
)


def refresh_training_state(state, run):
    """Observation only; unreadable/transient progress is never a restart signal."""
    try:
        progress = json.loads((Path(run) / "progress.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    state["model_training_started"] |= progress["model_training_started"]


def assert_condition_smoke(batch_root, expected_jobs):
    """Only actual completed, audited same-source smoke jobs qualify; no scores."""
    needed = {}
    for job in expected_jobs:
        path = Path(job["run"])
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        key = str(Path(config["bound_main_run"]["run"]).resolve())
        if key in needed:
            raise ValueError("duplicate expected condition model group")
        needed[key] = {"config": config, "run": path}
    if not needed:
        raise ValueError("condition smoke requires nonempty expected model groups")
    for candidate in sorted(Path(batch_root).glob("*/progress.json"), reverse=True):
        try:
            # A concurrently written or damaged historical observation cannot
            # authorize fitting, nor hide an older valid completed smoke.
            state = json.loads(candidate.read_text(encoding="utf-8"))
            if not state["complete"] or not state["model_training_started"]:
                continue
            batch_config = json.loads((candidate.parent / "config.json").read_text(encoding="utf-8"))
            if not batch_config["smoke"] or batch_config["prepare_only"] or (candidate.parent / "failure.json").exists():
                continue
            checked = {}
            for job in state["jobs_prepared"]:
                run = Path(job["run"])
                config = json.loads((run / "config.json").read_text(encoding="utf-8"))
                key = str(Path(config["bound_main_run"]["run"]).resolve())
                expected = needed[key]["config"]
                if key in checked or not config["smoke"]:
                    raise ValueError("duplicate/non-smoke condition job")
                for tree, field in (("source_snapshot", "source_sha256"),
                                    ("execution_snapshot", "execution_source_sha256")):
                    for name, digest in config[field].items():
                        relative = Path(name)
                        # Historical main sources sometimes recorded an
                        # absolute driver path and snapshot just its basename.
                        relative = relative.name if relative.is_absolute() else relative
                        source = run / tree / relative
                        if (not source.resolve().is_relative_to((run / tree).resolve())
                                or stream_sha(source) != digest):
                            raise ValueError("condition smoke source snapshot changed")
                for field in ("condition_plan_config_sha256", "condition_plan_splits_sha256",
                              "bound_main_run", "source_sha256", "execution_source_sha256"):
                    if config[field] != expected[field]:
                        raise ValueError("condition smoke source/plan contract differs from full job")
                models = config.get("architectures", list(config.get("model_candidates", {})))
                expected_models = expected.get("architectures", list(expected.get("model_candidates", {})))
                if models != expected_models or config["seeds"] != expected["seeds"][:1]:
                    raise ValueError("condition smoke model/seed selection differs")
                progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
                plans = json.loads((run / "splits.json").read_text(encoding="utf-8"))["plans"]
                full_plans = json.loads((needed[key]["run"] / "splits.json").read_text(encoding="utf-8"))["plans"]
                smoke_plans = [next(p for p in full_plans if p["dataset"] == domain) for domain in ("MIT", "XJTU")]
                grid = {(p["name"], seed, model) for p in smoke_plans for seed in config["seeds"] for model in models}
                completed = [(row["fold"], row["seed"], row["model"]) for row in progress["completed"]]
                if (plans != smoke_plans or not progress["complete"] or not progress["model_training_started"]
                        or len(completed) != len(set(completed)) or set(completed) != grid):
                    raise ValueError("condition smoke lacks full real fitted coverage")
                for fold, seed, model in completed:
                    unit = run / fold / str(seed) / model
                    audit = json.loads((unit / "artifact_audit.json").read_text(encoding="utf-8"))
                    if audit["status"] != "passed static artifact/metric audit" or not audit["artifact_sha256"]:
                        raise ValueError("condition smoke lacks numerical evidence")
                    required = {"config.json", "selected.json", "resources.json", "validation_scores.json",
                                str(Path("test/metrics.json")), str(Path("test/predictions.npz"))}
                    if "sequence_config" in config:
                        required.update(("model.pt", "model.json", "preprocessor.joblib"))
                    else:
                        required.add("model.joblib")
                    if required - set(audit["artifact_sha256"]):
                        raise ValueError("condition smoke fingerprints omit required model/prediction evidence")
                    for name, digest in audit["artifact_sha256"].items():
                        path = unit / name
                        if not path.resolve().is_relative_to(unit.resolve()) or stream_sha(path) != digest:
                            raise ValueError("condition smoke artifact changed")
                checked[key] = str(run.resolve())
            if set(checked) != set(needed):
                raise ValueError("condition smoke omitted a bound model group")
            return {"batch": str(candidate.parent.resolve()), "jobs": checked,
                "scope": "completed same-source engineering smoke; no accuracy threshold or test-driven selection"}
        except (ValueError, KeyError, OSError, StopIteration):
            continue
    raise ValueError("completed real condition smoke required before full condition training")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Save all job contracts without any fitting")
    args = parser.parse_args()
    plan_folder = args.plan or Path(json.loads(Path(
        "experiments/soh_research_v5/latest_condition_plan.json").read_text(encoding="utf-8"))["folder"])
    condition, frozen, _, _, plans = load_condition_contract(plan_folder)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    batch = Path("experiments/soh_research_v5/condition_batches") / timestamp
    batch.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(batch / "batch.log", encoding="utf-8"), logging.StreamHandler()])
    config = {"plan": str(plan_folder.resolve()), "plan_config_sha256": sha(plan_folder / "config.json"),
        "plan_splits_sha256": sha(plan_folder / "splits.json"), "smoke": args.smoke,
        "prepare_only": args.prepare_only, "model_training_started": False,
        "scope": "same physical families, complete known-condition holdout; separate from main grouped CV",
        "worker_order": "sequential isolated processes to avoid module/source and GPU competition",
        "main_bindings": condition["main_runs_with_frozen_model_search_and_source"],
        "source_sha256": {name: sha(WORKSPACE / name) for name in EXECUTION_SOURCES}}
    for name in EXECUTION_SOURCES:
        target = batch / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE / name, target)
    save_json(batch / "config.json", config)
    state = {"jobs_prepared": [], "jobs_finished": [], "complete": False, "model_training_started": False}
    save_json(batch / "progress.json", state)
    try:
        for index, binding in enumerate(config["main_bindings"]):
            source = Path(binding["run"])
            main_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
            job_config, selected_plans = condition_job_configuration(main_config, condition, plans, smoke=args.smoke)
            run = Path("experiments/soh_research_v5/condition_runs") / f"{timestamp}_group{index+1}"
            run.mkdir(parents=True, exist_ok=False)
            shutil.copytree(source / "source_snapshot", run / "source_snapshot")
            shutil.copytree(Path(condition["frozen_audit"]), run / "frozen_audit")
            shutil.copytree(plan_folder, run / "frozen_condition_plan")
            execution_hashes = {}
            for name in EXECUTION_SOURCES:
                destination = run / "execution_snapshot" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(WORKSPACE / name, destination)
                execution_hashes[name] = sha(destination)
            # Keep model source_sha256 exactly as the bound main run. Execution
            # code has its own independently verified snapshot namespace.
            job_config.update(condition_plan_config_sha256=sha(plan_folder / "config.json"),
                condition_plan_splits_sha256=sha(plan_folder / "splits.json"),
                main_bindings=config["main_bindings"], bound_main_run=binding,
                execution_source_sha256=execution_hashes, condition_batch=str(batch.resolve()),
                frozen_audit_config_sha256=sha(Path(condition["frozen_audit"]) / "config.json"),
                frozen_audit_files_sha256={str(p.relative_to(run / "frozen_audit")): sha(p)
                    for p in (run / "frozen_audit").rglob("*") if p.is_file()})
            save_json(run / "config.json", job_config)
            save_json(run / "splits.json", {"plans": selected_plans})
            models = job_config.get("architectures", list(job_config.get("model_candidates", {})))
            save_json(run / "progress.json", {"complete": False, "smoke": args.smoke,
                "completed": [], "summary": [], "model_training_started": False,
                "expected_units": len(selected_plans) * len(job_config["seeds"]) * len(models)})
            state["jobs_prepared"].append({"run": str(run.resolve()), "models": models,
                "expected_units": len(selected_plans) * len(job_config["seeds"]) * len(models),
                "token": f"condition-{timestamp}-group{index+1}"})
            save_json(batch / "progress.json", state)
        if args.prepare_only:
            save_json(batch / "prepared.json", {"prepared": True, "model_training_started": False,
                "jobs": state["jobs_prepared"]})
            print(json.dumps({"batch": str(batch.resolve()), "jobs": state["jobs_prepared"],
                "model_training_started": False}, indent=2))
            return
        gate = assert_main_complete(config["main_bindings"])
        numeric = assert_main_numerical_audits(config["main_bindings"],
            "experiments/soh_research_v5/artifact_audits", completion=gate)
        save_json(batch / "main_execution_gates.json", {"complete": gate, "numerical": numeric})
        if not args.smoke:
            save_json(batch / "condition_smoke_gate.json", assert_condition_smoke(
                WORKSPACE / "experiments/soh_research_v5/condition_batches", state["jobs_prepared"]))
        for job in state["jobs_prepared"]:
            # Worker verifies both gates again before its first fit.
            command = [sys.executable, str(Path(job["run"]) / "execution_snapshot/scripts/run_soh_research_condition_worker_v5.py"),
                "--run", job["run"], "--worker-token", job["token"]]
            logging.info("START CONDITION JOB %s", job["run"])
            process = subprocess.Popen(command, cwd=WORKSPACE)
            state["active_worker"] = {"pid": process.pid, "run": job["run"], "command": command}
            save_json(batch / "progress.json", state)
            while process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                refresh_training_state(state, job["run"])
                save_json(batch / "progress.json", state)
            refresh_training_state(state, job["run"])
            state["active_worker"] = None
            save_json(batch / "progress.json", state)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            outcome = json.loads((Path(job["run"]) / "progress.json").read_text(encoding="utf-8"))
            if not outcome["complete"]:
                raise ValueError("condition worker exited without full selected-grid completion")
            state["model_training_started"] |= outcome["model_training_started"]
            state["jobs_finished"].append(job["run"])
            save_json(batch / "progress.json", state)
        state["complete"] = True
        save_json(batch / "progress.json", state)
    except BaseException:
        for job in state["jobs_prepared"]:
            refresh_training_state(state, job["run"])
        save_json(batch / "progress.json", state)
        save_json(batch / "failure.json", {"traceback": traceback.format_exc(),
            "model_training_started": state["model_training_started"],
            "jobs_prepared": state["jobs_prepared"], "jobs_finished": state["jobs_finished"]})
        raise


if __name__ == "__main__":
    main()
