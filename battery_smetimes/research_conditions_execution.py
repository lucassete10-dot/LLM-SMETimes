"""Verify metadata-only condition plans and preserve the main search budgets."""

import copy
import json
from pathlib import Path

from battery_smetimes.physical_types import load_registry
from battery_smetimes.research_generalization_splits import make_condition_splits, metadata_only
from battery_smetimes.research_inference import verify_source_snapshot
from scripts.prepare_same_type_forecast import sha


def load_condition_contract(folder):
    folder = Path(folder)
    config = verify_source_snapshot(folder)
    audit = Path(config["frozen_audit"])
    if sha(audit / "config.json") != config["audit_config_sha256"]:
        raise ValueError("condition's original frozen audit changed")
    frozen = json.loads((audit / "config.json").read_text(encoding="utf-8"))
    if sha(audit / "cell_registry.json") != config["source_cell_registry_sha256"]:
        raise ValueError("condition cell registry changed")
    registered = json.loads((audit / "cell_registry.json").read_text(encoding="utf-8"))["cells"]
    registry_file = audit / "source_snapshot/battery_physical_registry.json"
    prepared = Path(json.loads((audit / "audit.json").read_text(encoding="utf-8"))["prepared_folder"])
    if (sha(registry_file) != config["registry_sha256"]
            or sha(prepared / "manifest.json") != config["prepared_manifest_sha256"]
            or sha(frozen["config"]["integrity_audit"]) != config["integrity_sha256"]):
        raise ValueError("condition registry/prepared manifest/integrity binding changed")
    registry = load_registry(registry_file)
    metadata = metadata_only(registered)
    if json.loads((folder / "metadata_only.json").read_text(encoding="utf-8"))["cells"] != metadata:
        raise ValueError("saved metadata-only condition cohort changed")
    plans = json.loads((folder / "splits.json").read_text(encoding="utf-8"))["plans"]
    if plans != make_condition_splits(metadata, frozen["config"], registry):
        raise ValueError("condition partitions differ from frozen metadata-only derivation")
    for field in ("origins_source_cycles", "training_seeds", "split_seed", "validation_fraction_of_development"):
        if config[field] != frozen["config"][field]:
            raise ValueError("condition seed/origin/validation contract changed")
    bindings = config["main_runs_with_frozen_model_search_and_source"]
    if not bindings or len({str(Path(b["run"]).resolve()) for b in bindings}) != len(bindings):
        raise ValueError("condition main-model bindings are missing or duplicated")
    for binding in bindings:
        run = Path(binding["run"])
        main = verify_source_snapshot(run)
        if (sha(run / "config.json") != binding["config_sha256"]
                or main["source_sha256"] != binding["source_sha256"]
                or sha(run / "frozen_audit/config.json") != config["audit_config_sha256"]):
            raise ValueError("bound main model source/configuration changed")
    return config, frozen, registered, registry, plans


def condition_job_configuration(main, condition, plans, *, smoke):
    """Different external evaluation splits, not a replacement main CV run."""
    if main["smoke"] or main["config"]["origins_source_cycles"] != condition["origins_source_cycles"]:
        raise ValueError("condition job needs a full main run with identical source origins")
    if main["seeds"] != condition["training_seeds"]:
        raise ValueError("condition job must preserve the complete frozen seed schedule")
    selected = list(plans)
    if smoke:
        selected = [next(p for p in plans if p["dataset"] == domain) for domain in ("MIT", "XJTU")]
    result = copy.deepcopy(main)
    result.update(smoke=smoke, folds=[p["name"] for p in selected],
        seeds=main["seeds"][:1] if smoke else list(main["seeds"]),
        evaluation_protocol="leave_condition_out_public_development_v1",
        evidence_level="engineering smoke, not full condition coverage" if smoke else "public development leave-condition-out",
        not_main_grouped_cv=True, selection_metric="non-heldout-condition validation cell-macro MAPE at origin100",
        global_summary_writer=False)
    if "sequence_config" not in main:
        candidates = main["model_candidates"] if "model_candidates" in main else main["config"]["initial_candidates"]
        result["model_candidates"] = copy.deepcopy(candidates)
    elif main["sequence_config"].get("kind") == "frozen_llm":
        if "llm_e1" not in main["architectures"]:
            raise ValueError("main SMETimes E1 absent from frozen neural source")
        result["architectures"] = ["llm_e1"]
        result["cross_condition_ablation_scope"] = "E1 compared with all bound small baselines; full LLM ablations remain the separate same-type main experiment"
    return result, selected


def verify_condition_model_contract(folder, config, recorded):
    """No fitting/deserialization: bind one completed unit to its true holdout.

    Static hashes are prerequisites for executable replay, not replay evidence.
    The separate main audit's grouped-CV splits must never be substituted for
    this model's condition holdout, or modified to make a loader accept it.
    """
    folder = Path(folder).resolve()
    run = folder.parents[2]
    if (config.get("evaluation_protocol") != "leave_condition_out_public_development_v1"
            or not config.get("not_main_grouped_cv") or config.get("global_summary_writer") is not False):
        raise ValueError("not an isolated condition experiment contract")
    for name, digest in config["execution_source_sha256"].items():
        path = run / "execution_snapshot" / name
        if (not path.resolve().is_relative_to((run / "execution_snapshot").resolve())
                or sha(path) != digest):
            raise ValueError("condition execution source changed")
    plan_folder = run / "frozen_condition_plan"
    if (sha(plan_folder / "config.json") != config["condition_plan_config_sha256"]
            or sha(plan_folder / "splits.json") != config["condition_plan_splits_sha256"]):
        raise ValueError("condition frozen plan fingerprint changed")
    condition, _, _, _, plans = load_condition_contract(plan_folder)
    binding = config["bound_main_run"]
    if (binding not in condition["main_runs_with_frozen_model_search_and_source"]
            or config["main_bindings"] != condition["main_runs_with_frozen_model_search_and_source"]):
        raise ValueError("condition model main source binding changed")
    main = json.loads((Path(binding["run"]) / "config.json").read_text(encoding="utf-8"))
    expected, selected = condition_job_configuration(main, condition, plans, smoke=config["smoke"])
    if any(config[name] != value for name, value in expected.items()):
        raise ValueError("condition model budget/source/seeds changed")
    if json.loads((run / "splits.json").read_text(encoding="utf-8"))["plans"] != selected:
        raise ValueError("condition model partition differs from metadata-only derivation")
    frozen_audit = run / "frozen_audit"
    if (sha(frozen_audit / "config.json") != config["frozen_audit_config_sha256"]
            or {str(p.relative_to(frozen_audit)): sha(p) for p in frozen_audit.rglob("*") if p.is_file()}
                != config["frozen_audit_files_sha256"]):
        raise ValueError("condition copied data audit changed")
    models = config.get("architectures", list(config.get("model_candidates", {})))
    plan = recorded["plan"]
    if (plan not in selected or recorded["seed"] not in config["seeds"] or recorded["model"] not in models
            or recorded["physical_family"] != plan["physical_family"]
            or folder != run / plan["name"] / str(recorded["seed"]) / recorded["model"]):
        raise ValueError("condition model identity/partition is not registered")
    candidates = (config["sequence_config"]["candidates"][:1] if config["smoke"] else
                  config["sequence_config"]["candidates"]) if "sequence_config" in config else config["model_candidates"][recorded["model"]]
    if recorded["candidates"] != candidates or recorded["smoke"] != config["smoke"]:
        raise ValueError("condition unit's declared search budget changed")
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    completed = [(r["fold"], r["seed"], r["model"]) for r in progress["completed"]]
    key = (plan["name"], recorded["seed"], recorded["model"])
    grid = {(p["name"], seed, model) for p in selected for seed in config["seeds"] for model in models}
    if (key not in completed or len(completed) != len(set(completed)) or set(completed) - grid
            or not progress["model_training_started"] or progress["expected_units"] != len(grid)):
        raise ValueError("condition model unit is not audited completed coverage")
    audit = json.loads((folder / "artifact_audit.json").read_text(encoding="utf-8"))
    required = {"config.json", "selected.json", "resources.json", "validation_scores.json",
                str(Path("test/metrics.json")), str(Path("test/predictions.npz"))}
    if "sequence_config" in config:
        required.update(("model.pt", "model.json", "preprocessor.joblib"))
        if config["sequence_config"]["kind"] == "frozen_llm":
            required.add("llm.json")
    else:
        required.add("model.joblib")
    if (audit["status"] != "passed static artifact/metric audit"
            or required - set(audit["artifact_sha256"])):
        raise ValueError("condition model missing its required static numerical audit")
    for name, digest in audit["artifact_sha256"].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder) or sha(path) != digest:
            raise ValueError("condition completed model/prediction artifacts changed")
    return {"scope": "one completed condition unit; executable replay remains separate",
        "dataset": plan["dataset"], "heldout_condition": plan["heldout_condition"],
        "physical_family": plan["physical_family"], "not_main_grouped_cv": True,
        "bound_main_run": binding, "static_audit_sha256": sha(folder / "artifact_audit.json"),
        "run_config_sha256": sha(run / "config.json"), "condition_splits_sha256": sha(run / "splits.json")}
