"""Complete shared-baseline provenance/numerical grid and real-smoke gates."""

import copy
import importlib.metadata
from pathlib import Path

from battery_smetimes.research_inference import verify_source_snapshot
from battery_smetimes.research_transfer_baseline_execution import audit_baseline_unit
from battery_smetimes.research_transfer_neural_audit import read
from battery_smetimes.research_transfer_vi120 import stream_sha


def verify_baseline_contract(run):
    run = Path(run)
    config = verify_source_snapshot(run)
    prepared = Path(config["prepared"])
    for name, field in (("config.json", "prepared_config_sha256"), ("manifest.json", "prepared_manifest_sha256"),
                        ("splits.json", "prepared_splits_sha256")):
        if stream_sha(prepared / name) != config[field]:
            raise ValueError("shared baseline prepared contract changed")
    inputs = verify_source_snapshot(prepared)
    from scripts.train_soh_research_transfer_baselines_v5 import derive_baseline_source, ORIGINAL_MODEL_FILES
    source = derive_baseline_source(config["main_run_bindings"], inputs, smoke=config["smoke"])
    original_snapshot = Path(source["binding"]["run"]) / "source_snapshot"
    original_hashes = {name: stream_sha(original_snapshot / name) for name in ORIGINAL_MODEL_FILES}
    seeds = inputs["training_seeds"][:1] if config["smoke"] else inputs["training_seeds"]
    plans = read(run / "splits.json")["directions"]
    if (config["baseline_source"] != source or config["seeds"] != seeds or config["original_model_sha256"] != original_hashes
            or any(config["source_sha256"].get(name) != digest for name, digest in original_hashes.items())
            or config["origins"] != inputs["origins_source_cycles"] or config["input_schema"] != inputs["input_schema"]
            or config["feature_names"] != inputs["feature_names"] or config["models"] != ["ridge", "extra_trees"]
            or config["device"] != "cpu" or config["target_calibration"] != [] or config["global_summary_writer"] is not False
            or not config["not_same_type_deployment"] or len(plans) != 2
            or plans != read(prepared / "splits.json")["directions"]):
        raise ValueError("shared baseline budget/seeds/schema/roles/research-only contract changed")
    for name, expected in config["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("shared baseline runtime package version changed")
    return config, plans


def audit_baseline_run(run, cells, *, require_complete=True):
    run = Path(run)
    config, plans = verify_baseline_contract(run)
    state = read(run / "progress.json")
    if (config["prepare_only"] or not state["model_training_started"] or (require_complete and not state["complete"])
            or (run / "failure.json").exists() or (run / "launcher_failure.json").exists()):
        raise ValueError("shared baseline run has not completed successful fitting")
    grid = {(p["name"], seed, model) for p in plans for seed in config["seeds"] for model in config["models"]}
    completed = [(r["direction"], r["seed"], r["model"]) for r in state["completed"]]
    if len(completed) != len(set(completed)) or set(completed) != grid or state["expected_final_model_units"] != len(grid):
        raise ValueError("shared baseline completed grid incomplete/duplicate/unknown")
    units = []
    for plan in plans:
        for seed in config["seeds"]:
            for model in config["models"]:
                unit = run / plan["name"] / str(seed) / model
                current = audit_baseline_unit(unit, config, plan, seed, model, cells)
                if read(unit / "static_unit_audit.json") != current:
                    raise ValueError("shared baseline unit static evidence changed")
                units.append({"direction": plan["name"], "seed": seed, "model": model,
                    "unit_audit_sha256": stream_sha(unit / "static_unit_audit.json"), "target_summary": current["target_summary"]})
    return {"schema": 1, "status": "passed complete shared baseline static run audit", "finalized_units": len(grid),
        "completed_grid": [list(key) for key in sorted(grid)], "units": units,
        "run_config_sha256": stream_sha(run / "config.json"), "splits_sha256": stream_sha(run / "splits.json"),
        "source_sha256": config["source_sha256"], "prepared_manifest_sha256": config["prepared_manifest_sha256"],
        "scope": "every candidate/source fold/refit/target static evidence; not executable replay or blind accuracy"}


def assert_baseline_smoke(root, expected, cells):
    """Both directions and models in same-source smoke, never a score filter."""
    for run in sorted(Path(root).iterdir(), reverse=True):
        try:
            config, _ = verify_baseline_contract(run)
            if not config["smoke"] or config["prepare_only"]:
                continue
            for name in ("prepared", "prepared_config_sha256", "prepared_manifest_sha256", "prepared_splits_sha256",
                         "main_run_bindings", "source_sha256", "original_model_sha256", "origins", "device", "models", "input_schema", "feature_names", "packages"):
                if config[name] != expected[name]:
                    raise ValueError("shared baseline smoke source/input configuration differs")
            source = copy.deepcopy(expected["baseline_source"])
            source["candidates"] = {model: candidates[:1] for model, candidates in source["candidates"].items()}
            source["tree_estimators"] = 48
            if config["baseline_source"] != source or config["seeds"] != expected["seeds"][:1]:
                raise ValueError("shared baseline smoke budget/seed differs")
            audit = audit_baseline_run(run, cells)
            if read(run / "static_run_audit.json") != audit or audit["finalized_units"] != 4:
                raise ValueError("shared baseline full four-unit smoke audit missing or changed")
            return {"run": str(run.resolve()), "config_sha256": stream_sha(run / "config.json"),
                "progress_sha256": stream_sha(run / "progress.json"), "static_run_audit_sha256": stream_sha(run / "static_run_audit.json"),
                "scope": "real complete same-source four-unit engineering smoke, no target-score threshold"}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise ValueError("completed audited same-source four-unit baseline smoke required before full training")
