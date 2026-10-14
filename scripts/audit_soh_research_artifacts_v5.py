"""Read-only audit of finalized predictions, selection, and artifact provenance.

This does not deserialize models, fit, select by test scores, delete artifacts,
or certify every saved model's executable replay. Replay coverage is separate.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import traceback

import numpy as np

from battery_smetimes.research_baselines import regression_metrics
from scripts.prepare_same_type_forecast import save_json
from scripts.train_same_type_forecast import load_data

WORKSPACE = Path(__file__).resolve().parents[1]


def copy_source_snapshot(source, output, workspace=WORKSPACE):
    source = Path(source)
    source = source.resolve() if source.is_absolute() else (workspace / source).resolve()
    relative = source.relative_to(workspace.resolve())
    destination = output / "source_snapshot" / relative
    if not destination.resolve().is_relative_to((output / "source_snapshot").resolve()) or destination.exists():
        raise ValueError("snapshot destination escapes its new audit or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def baseline_candidates(config):
    # The first immutable v5 driver stored these inside its frozen main config.
    # Never fill a missing historical field from today's defaults.
    return config["model_candidates"] if "model_candidates" in config else config["config"]["initial_candidates"]


def equal_number(actual, expected, context):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{context}: missing-number mismatch")
    elif not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-9, atol=1e-10):
        raise ValueError(f"{context}: saved number differs from prediction arrays")


def equal_metrics(actual, expected, context):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{context}: unavailable-horizon mismatch")
        return
    for name, value in expected.items():
        equal_number(actual[name], value, f"{context}/{name}")


def audit_scores(folder, ids, origins, family, cells=None):
    """Account for every cell/origin, including skips, and recompute all metrics."""
    report = read_json(folder / "metrics.json")
    expected, accounted, array_names = {(cell, origin) for cell in ids for origin in origins}, set(), set()
    with np.load(folder / "predictions.npz", allow_pickle=False) as arrays:
        for row in report["by_cell"]:
            key = (row["cell_id"], row["origin"])
            if key not in expected or key in accounted or row["physical_family"] != family:
                raise ValueError("invalid or duplicate prediction cell/origin/family membership")
            accounted.add(key)
            prefix = f"{key[0]}__{key[1]}__"
            names = [prefix + suffix for suffix in ("cycle", "truth", "prediction")]
            array_names.update(names)
            cycle, truth, prediction = [arrays[name] for name in names]
            if any(value.ndim != 1 or len(value) != row["future_targets"] or not np.isfinite(value).all()
                   for value in (cycle, truth, prediction)) or len(cycle) < 2:
                raise ValueError("invalid prediction array shape, count, or finite values")
            if np.any(np.diff(cycle) <= 0) or np.any(cycle <= key[1]) or np.any(truth <= 0):
                raise ValueError("invalid future target grid or truth")
            if cells is not None:
                original = cells[key[0]]
                take = np.isfinite(original["soh"]) & (original["soh"] > 0) & (original["label_cycle"] > key[1])
                if not np.array_equal(cycle, original["label_cycle"][take]) or not np.array_equal(truth, original["soh"][take]):
                    raise ValueError("saved truth/grid differs from audited prepared targets")
            equal_metrics(row["metrics"], regression_metrics(truth, prediction), f"{key}/all_future")
            for horizon in (10, 50, 100):
                take = cycle <= key[1] + horizon
                metric = regression_metrics(truth[take], prediction[take]) if take.any() else None
                equal_metrics(row["within_horizon"][str(horizon)], metric, f"{key}/within_{horizon}")
            take = cycle >= (cycle[0] + cycle[-1]) / 2
            equal_metrics(row["late_half"], regression_metrics(truth[take], prediction[take]), f"{key}/late_half")
        if set(arrays.files) != array_names:
            raise ValueError("unreported or missing prediction arrays")
    for row in report["skipped"]:
        key = (row["cell_id"], row["origin"])
        if key not in expected or key in accounted or not row["reason"]:
            raise ValueError("invalid or duplicate skipped cell/origin")
        accounted.add(key)
    if accounted != expected:
        raise ValueError("some expected cell/origin pairs disappeared without coverage reasons")
    for origin in origins:
        values = [row["metrics"]["mape_percent"] for row in report["by_cell"] if row["origin"] == origin]
        summary = report["summary"][str(origin)]
        if summary["cells"] != len(values):
            raise ValueError("summary battery count differs from prediction records")
        equal_number(summary["cell_macro_mape_percent"], float(np.mean(values)) if values else None, "cell-macro summary")
        equal_number(summary["fraction_cells_mape_le_6"], float(np.mean(np.asarray(values) <= 6)) if values else None, "target-attainment summary")
    return report


def audit_selection(folder, candidates, validation_reports):
    scores, selected = read_json(folder / "validation_scores.json"), read_json(folder / "selected.json")
    if len(scores) != len(candidates) or [row["index"] for row in scores] != list(range(len(candidates))):
        raise ValueError("candidate search budget/index differs from frozen settings")
    for index, (row, parameters, report) in enumerate(zip(scores, candidates, validation_reports)):
        if row["parameters"] != parameters:
            raise ValueError("candidate parameters differ from frozen settings")
        value = report["summary"]["100"]["cell_macro_mape_percent"]
        if value is None:
            raise ValueError("candidate has no primary-origin validation score")
        equal_number(row["validation_mape_percent"], value, f"candidate {index} validation criterion")
    if selected != min(scores, key=lambda row: row["validation_mape_percent"]):
        raise ValueError("saved selection is not the minimum validation-only criterion")
    return selected


def collect_replays(root):
    records = {}
    for path in sorted(root.glob("*/report.json")):
        report, config = read_json(path), read_json(path.parent / "config.json")
        folder = str(Path(config["model_folder"]).resolve())
        records.setdefault(folder, []).append({"report_sha256": fingerprint(path),
            "path": str(path.resolve()), "artifact_sha256": report["evidence"]["artifact_sha256"],
            "no_fit": report["evidence"].get("no_fit", False),
            "passed_rows": [row for row in report["rows"] if row.get("passed")]})
    return records


def collect_replay_failures(root):
    records = {}
    for path in sorted(root.glob("*/failure.json")):
        config_path = path.parent / "config.json"
        if not config_path.is_file():
            continue
        config, failure = read_json(config_path), read_json(path)
        records.setdefault(str(Path(config["model_folder"]).resolve()), []).append({
            "path": str(path.resolve()), "failure_sha256": fingerprint(path),
            "config_sha256": fingerprint(config_path), "device": config["device"],
            "original_training_device": config.get("original_training_device"),
            "failed_rows": [row for row in failure.get("rows", []) if row.get("passed") is False],
            "traceback": failure["traceback"]})
    return records


def audit_unit(folder, run_config, plan, seed, model, registered, cells, replays, registry, replay_failures=None):
    config = read_json(folder / "config.json")
    family = registered[plan["train"][0]]["physical_family"]
    if config["plan"] != plan or config["seed"] != seed or config["model"] != model or config["physical_family"] != family:
        raise ValueError("saved unit identity differs from frozen plan")
    members = plan["train"] + plan["validation"] + plan["test"]
    if len(set(members)) != len(members) or any(registered[cell]["physical_family"] != family for cell in members):
        raise ValueError("overlapping partitions or incompatible registered physical families")
    neural = "sequence_config" in run_config
    candidates = run_config["sequence_config"]["candidates"] if neural else baseline_candidates(run_config)[model]
    if config["candidates"] != candidates or config["smoke"] != run_config["smoke"]:
        raise ValueError("saved candidate search configuration changed")
    origins = run_config["config"]["origins_source_cycles"]
    validation_reports = []
    files = [folder / name for name in ("config.json", "selected.json", "validation_scores.json", "resources.json")]
    for index in range(len(candidates)):
        candidate = folder / f"candidate_{index}"
        if neural:
            info = read_json(candidate / "model.json")
            if (info["train_ids"] != plan["train"] or info["parameters"] != candidates[index] or info["seed"] != seed
                    or info["physical_family"] != family or info["registry"] != registry
                    or info["origins"] != origins or info["architecture"] != model):
                raise ValueError("candidate model metadata contradicts train-only fitting")
            files.extend(candidate / name for name in ("model.pt", "model.json", "preprocessor.joblib"))
        else:
            files.append(candidate / "model.joblib")
        validation_reports.append(audit_scores(candidate / "validation", plan["validation"], origins, family, cells))
        files.extend(candidate / "validation" / name for name in ("metrics.json", "predictions.npz"))
    selected = audit_selection(folder, candidates, validation_reports)
    if neural:
        info = read_json(folder / "model.json")
        if (info["train_ids"] != plan["train"] + plan["validation"] or info["epochs"] != selected["selected_epochs"]
                or info["parameters"] != selected["parameters"] or info["physical_family"] != family or info["seed"] != seed
                or info["registry"] != registry or info["origins"] != origins or info["architecture"] != model):
            raise ValueError("final neural metadata contradicts the frozen validation selection/refit")
        files.extend(folder / name for name in ("model.pt", "model.json", "preprocessor.joblib"))
        if (folder / "llm.json").is_file():
            llm = read_json(folder / "llm.json")
            if (llm["variant"] != run_config["sequence_config"]["variants"][model]
                    or llm["checkpoint_files_sha256"] != run_config["checkpoint_files_sha256"]):
                raise ValueError("LLM variant or checkpoint fingerprint differs from frozen run")
            files.append(folder / "llm.json")
        elif run_config["sequence_config"].get("kind") == "frozen_llm":
            raise ValueError("frozen LLM model lacks its backbone provenance")
    else:
        files.append(folder / "model.joblib")
    test = audit_scores(folder / "test", plan["test"], origins, family, cells)
    files.extend(folder / "test" / name for name in ("metrics.json", "predictions.npz", "curves_origin100.png"))
    hashes = {}
    for path in files:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing/empty finalized artifact: {path.relative_to(folder)}")
        hashes[str(path.relative_to(folder))] = fingerprint(path)
    valid_replays = []
    for replay in replays.get(str(folder.resolve()), []):
        if not replay["no_fit"] or any(hashes.get(name) != value for name, value in replay["artifact_sha256"].items()):
            raise ValueError("saved executable replay points to different current model artifacts")
        valid_replays.append(replay)
    return {"status": "passed static artifact/metric audit", "artifact_sha256": hashes,
        "validation_batteries": len(plan["validation"]), "test_batteries": len(plan["test"]),
        "reported_test_cell_origin_pairs": len(test["by_cell"]), "skipped_test_cell_origin_pairs": len(test["skipped"]),
        "executable_replays": valid_replays,
        "known_executable_replay_failures": (replay_failures or {}).get(str(folder.resolve()), []),
        "executable_replay_checked_pairs": sorted({(row["cell_id"], row["origin"]) for replay in valid_replays for row in replay["passed_rows"]}),
        "scope": "All stored metrics/target grids and validation selection checked; weight hashes recorded, not deserialized or executed by this audit."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    args = parser.parse_args()
    output = Path("experiments/soh_research_v5/artifact_audits") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path("battery_smetimes/research_baselines.py"), Path("scripts/train_same_type_forecast.py"),
               Path("scripts/prepare_same_type_forecast.py")]
    try:
        for source in sources:
            copy_source_snapshot(source, output)
    except Exception:
        save_json(output / "failure.json", {"stage": "auditor source snapshot", "traceback": traceback.format_exc()})
        raise
    save_json(output / "config.json", {"runs": [str(path.resolve()) for path in args.run], "numpy": np.__version__,
        "audit_contract_schema_version": 2,
        "source_sha256": {str(path): fingerprint(path) for path in sources},
        "policy": "Read-only originals; finalized units from initial progress snapshot only; no fitting, deletion or test-dependent selection; hashes do not substitute for executable replay."})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "audit.log", encoding="utf-8"), logging.StreamHandler()])
    rows, runs, data_cache = [], [], {}
    replays = collect_replays(Path("experiments/soh_research_v5/replays"))
    replay_failures = collect_replay_failures(Path("experiments/soh_research_v5/replays"))
    for run in args.run:
        try:
            config = read_json(run / "config.json")
            progress_bytes = (run / "progress.json").read_bytes()
            progress = json.loads(progress_bytes)
            if config["smoke"]:
                raise ValueError("smoke run cannot be presented as full-research artifact coverage")
            source_hashes = {}
            for name, expected_hash in config["source_sha256"].items():
                original = Path(name)
                path = run / "source_snapshot" / (original.name if original.is_absolute() else original)
                if not path.resolve().is_relative_to((run / "source_snapshot").resolve()):
                    raise ValueError("snapshot source path escapes the experiment")
                if fingerprint(path) != expected_hash:
                    raise ValueError("frozen model source changed")
                source_hashes[name] = expected_hash
            frozen = read_json(run / "frozen_audit/config.json")
            registry_path = run / "frozen_audit/source_snapshot/battery_physical_registry.json"
            if fingerprint(registry_path) != frozen["registry_sha256"]:
                raise ValueError("frozen physical registry changed")
            registry = read_json(registry_path)
            if config.get("sequence_config", {}).get("kind") == "frozen_llm":
                checkpoint = Path(config["sequence_config"]["checkpoint"])
                for name, expected_hash in config["checkpoint_files_sha256"].items():
                    if fingerprint(checkpoint / name) != expected_hash:
                        raise ValueError("frozen LLM checkpoint changed")
            prepared = Path(read_json(run / "frozen_audit/audit.json")["prepared_folder"])
            integrity = Path(frozen["config"]["integrity_audit"])
            if fingerprint(prepared / "manifest.json") != frozen["prepared_manifest_sha256"] or fingerprint(integrity) != frozen["integrity_sha256"]:
                raise ValueError("prepared manifest or frozen integrity rules changed")
            data_key = (str(prepared.resolve()), frozen["prepared_manifest_sha256"], frozen["integrity_sha256"])
            if data_key not in data_cache:
                data_cache[data_key] = load_data(prepared, integrity)
            manifest, cells = data_cache[data_key]
            if config["prepared_files_sha256"] != {row["cell_id"]: row["sha256"] for row in manifest["cells"]}:
                raise ValueError("prepared cell fingerprints differ from saved experiment")
            plans = read_json(run / "splits.json")["plans"]
            frozen_plans = read_json(run / "frozen_audit/splits.json")["plans"]
            if any(plan not in frozen_plans for plan in plans):
                raise ValueError("run plans differ from frozen whole-cell audit")
            registered = read_json(run / "frozen_audit/cell_registry.json")["cells"]
            models = config["architectures"] if "sequence_config" in config else list(baseline_candidates(config))
            expected = {(plan["name"], seed, model) for plan in plans for seed in config["seeds"] for model in models}
            completed = [(row["fold"], row["seed"], row["model"]) for row in progress["completed"]]
            if len(set(completed)) != len(completed) or set(completed) - expected or (progress["complete"] and set(completed) != expected):
                raise ValueError("progress membership/terminal flag contradicts frozen run grid")
            runs.append({"run": str(run.resolve()), "expected_units": len(expected), "snapshot_completed_units": len(completed),
                "run_config_sha256": fingerprint(run / "config.json"),
                "frozen_audit_sha256": {str(p.relative_to(run / "frozen_audit")): fingerprint(p)
                    for p in sorted((run / "frozen_audit").rglob("*")) if p.is_file()},
                "run_claimed_complete": progress["complete"], "pending_units": sorted(expected - set(completed)),
                "frozen_source_sha256": source_hashes, "progress_sha256_at_snapshot": hashlib.sha256(progress_bytes).hexdigest(),
                "top_level_failure_present": (run / "failure.json").exists()})
            for fold, seed, model in completed:
                row = {"run": run.name, "fold": fold, "seed": seed, "model": model}
                try:
                    plan = next(plan for plan in plans if plan["name"] == fold)
                    row.update(audit_unit(run / fold / str(seed) / model, config, plan, seed, model, registered, cells, replays, registry, replay_failures))
                except Exception:
                    row.update(status="failed artifact audit", traceback=traceback.format_exc())
                rows.append(row)
                if len(rows) % 10 == 0:
                    logging.info("AUDITED %s finalized units; failures=%s", len(rows), sum("failed" in row["status"] for row in rows))
                    save_json(output / "progress.json", {"runs": runs, "units": rows})
        except Exception:
            runs.append({"run": str(run.resolve()), "status": "failed run provenance audit", "traceback": traceback.format_exc()})
    report = {"runs": runs, "units": rows, "failed_units": sum("failed" in row["status"] for row in rows),
        "audit_contract_schema_version": 2,
        "failed_runs": sum("failed" in row.get("status", "") for row in runs),
        "known_failed_executable_replays_in_audited_units": sum(len(row.get("known_executable_replay_failures", [])) for row in rows),
        "scope": "Snapshot of finalized units only. Pending units are not failures or verified artifacts. Prepared-cell hashes verified; raw MAT extraction is not rerun. Static audit is not whole-model executable replay or full-goal completion."}
    save_json(output / "report.json", report)
    save_json(Path("experiments/soh_research_v5/latest_artifact_audit.json"), {"folder": str(output.resolve())})
    print(json.dumps({"folder": str(output.resolve()), "audited_units": len(rows), "failed_units": report["failed_units"], "failed_runs": report["failed_runs"]}, indent=2))
    if report["failed_units"] or report["failed_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
