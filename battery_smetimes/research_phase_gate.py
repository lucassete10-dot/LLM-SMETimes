"""Machine-check the prespecified main-before-generalization execution order."""

import json
from pathlib import Path

from battery_smetimes.research_inference import verify_source_snapshot
from scripts.prepare_same_type_forecast import sha
from battery_smetimes.research_transfer_vi120 import stream_sha


REQUIRED_MAIN_MODELS = {
    "ridge", "extra_trees", "elastic_net", "gpr", "lightgbm",
    "gru", "lstm", "tcn", "transformer", "statistics_mlp",
    "llm_e1", "llm_e2", "llm_e4", "llm_experts1", "llm_experts8", "llm_matched_plain",
}


def main_run_bindings(root, audit_config_sha256):
    rows = []
    for run in sorted(Path(root).glob("*")):
        if not (run / "config.json").is_file() or not (run / "progress.json").is_file():
            continue
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if config["smoke"]:
            continue
        if sha(run / "frozen_audit/config.json") != audit_config_sha256:
            raise ValueError("main-run binding refers to a different frozen audit")
        rows.append({"run": str(run.resolve()), "config_sha256": sha(run / "config.json")})
    return rows


def assert_main_complete(bindings):
    """Do not mistake count-only progress or a smoke run for full completion."""
    observed_models, reports = set(), []
    for binding in bindings:
        run = Path(binding["run"])
        if sha(run / "config.json") != binding["config_sha256"]:
            raise ValueError("bound main configuration changed")
        config = verify_source_snapshot(run)
        if config["smoke"]:
            raise ValueError("smoke runs cannot satisfy the main-completion gate")
        frozen = json.loads((run / "frozen_audit/config.json").read_text(encoding="utf-8"))["config"]
        plans = json.loads((run / "frozen_audit/splits.json").read_text(encoding="utf-8"))["plans"]
        folds = [plan["name"] for plan in plans]
        if not config["source_sha256"]:
            raise ValueError("main source snapshot cannot be empty")
        if len(config["folds"]) != len(set(config["folds"])) or set(config["folds"]) != set(folds):
            raise ValueError("subset of frozen outer folds cannot satisfy main completion")
        if len(frozen["training_seeds"]) < 5 or len(config["seeds"]) != len(set(config["seeds"])) or set(config["seeds"]) != set(frozen["training_seeds"]):
            raise ValueError("all frozen training seeds are required before generalization")
        candidates = config.get("model_candidates", config["config"].get("initial_candidates", {}))
        models = config.get("architectures", list(candidates))
        expected = {(fold, seed, model) for fold in config["folds"]
                    for seed in config["seeds"] for model in models}
        progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
        completed = [(r["fold"], r["seed"], r["model"]) for r in progress["completed"]]
        if not expected or not progress["complete"] or len(completed) != len(set(completed)) or set(completed) != expected:
            raise ValueError(f"same-type main experiment incomplete: {run.name}; {len(completed)}/{len(expected)} units")
        if (run / "failure.json").is_file():
            raise ValueError("main run has an unresolved top-level failure")
        for fold, seed, model in completed:
            folder = run / fold / str(seed) / model
            for name in ("resources.json", "test/metrics.json", "test/predictions.npz"):
                if not (folder / name).is_file():
                    raise ValueError("main completion claims an unfinalized unit")
        observed_models.update(models)
        reports.append({"run": str(run.resolve()), "progress_sha256": sha(run / "progress.json"),
                        "finalized_units": len(completed),
                        "scope": "execution-order/source-snapshot/finalization guard; not a replacement for metric or full-goal audit"})
    if REQUIRED_MAIN_MODELS - observed_models:
        raise ValueError("required main models/ablations missing from the completion gate")
    return reports


def select_source_candidate(candidates, expected_source_validation_ids, origin=100):
    """Only source-validation errors are admissible; no score-based exclusions."""
    import numpy as np
    expected = set(expected_source_validation_ids)
    if not candidates or not expected:
        raise ValueError("candidate selection needs nonempty source validation")
    scores = []
    for candidate in candidates:
        if any(row["cell_id"] not in expected for row in candidate["by_cell"]):
            raise ValueError("non-source cell entered source candidate selection")
        rows = [row for row in candidate["by_cell"] if row["origin"] == origin]
        ids = [row["cell_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError("source validation must cover each required cell exactly once")
        values = [row["metrics"]["mape_percent"] for row in rows]
        if not np.isfinite(values).all() or any(value < 0 for value in values):
            raise ValueError("invalid source error cannot be silently filtered")
        scores.append({"index": candidate["index"], "parameters": candidate["parameters"],
            "source_validation_cell_macro_mape_percent": float(np.mean(values)),
            "source_validation_cells": len(expected)})
    if len({row["index"] for row in scores}) != len(scores):
        raise ValueError("duplicate source candidate index")
    return min(scores, key=lambda row: row["source_validation_cell_macro_mape_percent"]), scores


def assert_main_numerical_audits(bindings, audit_root, *, completion=None):
    """Bind full numerical audits to unchanged run, data-rule and model files.

    This consumes the separate auditor's evidence; it does not execute weights
    or replace the required executable-replay and scientific conclusion steps.
    Older unbound reports remain historical evidence, not this execution gate.
    """
    current_completion = assert_main_complete(bindings)
    if completion is not None and completion != current_completion:
        raise ValueError("numerical gate completion evidence is stale")
    completion = current_completion
    if {str(Path(row["run"]).resolve()) for row in completion} != {str(Path(row["run"]).resolve()) for row in bindings}:
        raise ValueError("numerical gate completion evidence does not cover bound main runs")
    workspace = Path(__file__).resolve().parents[1]
    candidates = sorted(Path(audit_root).glob("*/report.json"), reverse=True)
    evidence = []
    for binding in bindings:
        run = Path(binding["run"]).resolve()
        config = verify_source_snapshot(run)
        if stream_sha(run / "config.json") != binding["config_sha256"]:
            raise ValueError("bound main configuration changed")
        progress_hash = stream_sha(run / "progress.json")
        progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
        expected = {(row["fold"], row["seed"], row["model"]) for row in progress["completed"]}
        frozen_hashes = {str(p.relative_to(run / "frozen_audit")): stream_sha(p)
            for p in sorted((run / "frozen_audit").rglob("*")) if p.is_file()}
        matched, reasons = None, []
        for report_path in candidates:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            selected = [row for row in report["runs"] if Path(row["run"]).resolve() == run]
            if not selected:
                continue
            try:
                if report.get("audit_contract_schema_version") != 2 or report["failed_units"] or report["failed_runs"] or len(selected) != 1:
                    raise ValueError("audit is old/unbound, failed or has duplicate run records")
                audited = selected[0]
                if (not audited["run_claimed_complete"] or audited["pending_units"] or audited["top_level_failure_present"]
                        or audited["expected_units"] != len(expected) or audited["snapshot_completed_units"] != len(expected)):
                    raise ValueError("numerical audit did not cover the entire finalized run")
                if (audited["run_config_sha256"] != binding["config_sha256"]
                        or audited["progress_sha256_at_snapshot"] != progress_hash
                        or audited["frozen_source_sha256"] != config["source_sha256"]
                        or audited["frozen_audit_sha256"] != frozen_hashes):
                    raise ValueError("numerical audit run/progress/source/frozen-data binding is stale")
                audit_config = json.loads((report_path.parent / "config.json").read_text(encoding="utf-8"))
                if audit_config.get("audit_contract_schema_version") != 2 or not audit_config["source_sha256"]:
                    raise ValueError("missing bound numerical-auditor source snapshot")
                for name, digest in audit_config["source_sha256"].items():
                    original = Path(name)
                    relative = original.resolve().relative_to(workspace) if original.is_absolute() else original
                    snapshot = report_path.parent / "source_snapshot" / relative
                    if not snapshot.resolve().is_relative_to((report_path.parent / "source_snapshot").resolve()) or stream_sha(snapshot) != digest:
                        raise ValueError("numerical-auditor source snapshot changed")
                units = [row for row in report["units"] if row["run"] == run.name]
                keys = [(row["fold"], row["seed"], row["model"]) for row in units]
                if len(keys) != len(set(keys)) or set(keys) != expected:
                    raise ValueError("numerical audit omits or duplicates a main unit")
                for row in units:
                    if row["status"] != "passed static artifact/metric audit" or not row["artifact_sha256"]:
                        raise ValueError("main unit lacks successful numerical artifact evidence")
                    unit = run / row["fold"] / str(row["seed"]) / row["model"]
                    required = {"config.json", "selected.json", "resources.json",
                                "validation_scores.json", str(Path("test/metrics.json")),
                                str(Path("test/predictions.npz")), str(Path("test/curves_origin100.png"))}
                    neural = "sequence_config" in config
                    required.update(("model.pt", "model.json", "preprocessor.joblib") if neural else ("model.joblib",))
                    settings = (config["sequence_config"]["candidates"] if neural else
                        config.get("model_candidates", config["config"].get("initial_candidates", {}))[row["model"]])
                    for index in range(len(settings)):
                        candidate = Path(f"candidate_{index}")
                        required.update(str(candidate / name) for name in (
                            ("model.pt", "model.json", "preprocessor.joblib") if neural else ("model.joblib",)))
                        required.update(str(candidate / "validation" / name) for name in ("metrics.json", "predictions.npz"))
                    if config.get("sequence_config", {}).get("kind") == "frozen_llm":
                        required.add("llm.json")
                    if required - set(row["artifact_sha256"]):
                        raise ValueError("numerical audit lacks required unit fingerprints")
                    for name, digest in row["artifact_sha256"].items():
                        path = unit / name
                        if not path.resolve().is_relative_to(unit.resolve()) or stream_sha(path) != digest:
                            raise ValueError("numerically audited model/prediction artifact changed")
                matched = {"run": str(run), "report": str(report_path.resolve()),
                    "report_sha256": stream_sha(report_path), "audited_units": len(units),
                    "scope": "current immutable full numerical-audit evidence; not executable weight replay"}
                break
            except (ValueError, KeyError, OSError) as error:
                reasons.append(str(error))
        if matched is None:
            raise ValueError(f"current full numerical audit required before generalization: {run.name}; "
                             + "; ".join(dict.fromkeys(reasons)))
        evidence.append(matched)
    return evidence
