"""Independent complete paired shared-input tables; no fitting/weight loading."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from battery_smetimes.research_inference import verify_source_snapshot
from battery_smetimes.research_phase_gate import assert_main_complete, assert_main_numerical_audits
from battery_smetimes.research_transfer_baseline_audit import audit_baseline_run
from battery_smetimes.research_transfer_neural_audit import audit_shared_neural_run, read
from battery_smetimes.research_transfer_reporting import build_transfer_tables, validate_comparison_contracts, METRICS
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.train_soh_research_transfer_v5 import load_shared_cells
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_soh_research_shared_transfer_v5 import SOURCES as AUDIT_SOURCES


SOURCES = tuple(dict.fromkeys((*AUDIT_SOURCES, "scripts/report_soh_research_shared_v5.py",
    "battery_smetimes/research_transfer_reporting.py")))


def read_completed_run(run, protocol, workspace):
    run = Path(run).resolve()
    root = workspace / f"experiments/soh_research_v5/transfer_{protocol}_runs"
    if not run.is_relative_to(root) or run == root:
        raise ValueError("report requires explicit matching trusted-local baseline/neural roots")
    config = verify_source_snapshot(run)
    state = read(run / "progress.json")
    if (config["prepare_only"] or config["smoke"] or not state["complete"] or not state["model_training_started"]
            or (run / "failure.json").exists() or (run / "launcher_failure.json").exists()):
        raise ValueError("formal comparison requires successful complete real fitting; no smoke/prepare/partial/failed accuracy")
    for name, version in config["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("report original runtime package versions changed")
    return config, read(run / "splits.json")["directions"]


def read_audited_units(run, protocol, config, plans, expected_unit_hashes):
    """Read only files whose bytes still match freshly checked whole-unit audit."""
    units, provenance = {}, []
    for plan in plans:
        for seed in config["seeds"]:
            for model in config["models"]:
                unit = run / plan["name"] / str(seed) / model
                audit_sha = stream_sha(unit / "static_unit_audit.json")
                if audit_sha != expected_unit_hashes[(plan["name"], seed, model)]:
                    raise ValueError("unit audit changed after complete-run verification")
                audit = read(unit / "static_unit_audit.json")
                paths = {name: unit / "target_test" / name for name in ("metrics.json", "predictions.npz")}
                for name, path in paths.items():
                    if stream_sha(path) != audit["artifact_sha256"][str(path.relative_to(unit))]:
                        raise ValueError("audited target evidence changed before aggregation")
                metrics = read(paths["metrics.json"])
                with np.load(paths["predictions.npz"], allow_pickle=False) as saved:
                    arrays = {key: saved[key].copy() for key in saved.files}
                for name, path in paths.items():
                    if stream_sha(path) != audit["artifact_sha256"][str(path.relative_to(unit))]:
                        raise ValueError("target evidence changed while reading")
                if stream_sha(unit / "static_unit_audit.json") != audit_sha:
                    raise ValueError("unit audit changed while reading")
                units[(protocol, plan["name"], seed, model)] = {"metrics": metrics, "arrays": arrays}
                provenance.append({"protocol": protocol, "direction": plan["name"], "seed": seed, "model": model,
                    "unit_audit_sha256": audit_sha,
                    "metrics_sha256": stream_sha(paths["metrics.json"]), "predictions_sha256": stream_sha(paths["predictions.npz"])})
    return units, provenance


def write_tables(output, report):
    summaries, cells = [], []
    for section in ("primary", "matched_prefix_and_future"):
        for row in report[section]:
            identity = {key: row[key] for key in ("direction", "source_dataset", "target_dataset", "source_physical_family",
                "target_physical_family", "protocol", "model", "origin_source_cycle", "horizon", "evidence_level")}
            summary = {"table": section, **identity, "full_target_cells": row["full_target_cells"],
                "evaluation_cells": row["evaluation_cells"], "coverage_fraction": row["coverage_fraction"],
                "unavailable_reason": row.get("unavailable_reason"), "not_same_type_deployment": True,
                "fraction_cells_seed_averaged_mape_le_6": row.get("fraction_cells_seed_averaged_mape_le_6")}
            for metric in METRICS:
                values = row.get("metrics", {}).get(metric, {}).get("seed_macro_mean_and_sd", {})
                summary.update({metric+"_mean": values.get("mean"), metric+"_seed_sd": values.get("sd"), metric+"_n_defined_seeds": values.get("n_defined", 0)})
                summary[metric+"_undefined_cell_seed_pairs"] = row.get("metrics", {}).get(metric, {}).get("undefined_cell_seed_pairs")
            summaries.append(summary)
            for cell in row.get("by_cell_training_seed_variation", []):
                record = {"table": section, **identity, "cell_id": cell["cell_id"]}
                for metric in METRICS:
                    record.update({metric+"_mean": cell[metric]["mean"], metric+"_seed_sd": cell[metric]["sd"], metric+"_n_defined_seeds": cell[metric]["n_defined"]})
                cells.append(record)
    for name, records in (("summary.csv", summaries), ("by_cell.csv", cells)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["no_results"])
            writer.writeheader()
            writer.writerows(records)
    save_json(output / "coverage.json", report["coverage"])
    return {"summary_rows": len(summaries), "cell_rows": len(cells)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--neural-run", required=True, type=Path)
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    output = workspace / "results/soh_research_v5_shared" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "report.log", encoding="utf-8"), logging.StreamHandler()])
    hashes = {}
    observer_root = Path(__file__).resolve().parents[1]
    for name in SOURCES:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(observer_root / name, destination)
        hashes[name] = stream_sha(destination)
    runs = {"baseline": args.baseline_run.resolve(), "neural": args.neural_run.resolve()}
    save_json(output / "config.json", {"runs": {key: str(run) for key, run in runs.items()}, "source_sha256": hashes,
        "observer_scope": "independent current read-only observer, not original model execution",
        "python": sys.version, "numpy": np.__version__,
        "weights_loaded": False, "training_performed": False, "global_summary_writer": False,
        "target_calibration": [], "predictive_weight_replay_certified_by_this_report": False})
    try:
        configs, plans = {}, {}
        for protocol, run in runs.items():
            configs[protocol], plans[protocol] = read_completed_run(run, protocol, workspace)
        validate_comparison_contracts(configs, plans)
        completion = assert_main_complete(configs["baseline"]["main_run_bindings"])
        numerical = assert_main_numerical_audits(configs["baseline"]["main_run_bindings"], workspace / "experiments/soh_research_v5/artifact_audits", completion=completion)
        cells = load_shared_cells(Path(configs["baseline"]["prepared"]), progress_callback=lambda key, count, total:
            logging.info("REPORT SHARED INPUT %s %s/%s", key, count, total))
        evidence, units, provenance = {}, {}, []
        for protocol, run in runs.items():
            static = (audit_baseline_run if protocol == "baseline" else audit_shared_neural_run)(run, cells)
            if static != read(run / "static_run_audit.json"):
                raise ValueError("formal report parent static run evidence changed")
            evidence[protocol] = {"run": str(run), "config_sha256": stream_sha(run / "config.json"),
                "progress_sha256": stream_sha(run / "progress.json"), "static_run_audit_sha256": stream_sha(run / "static_run_audit.json"),
                "finalized_units": static["finalized_units"]}
            expected_hashes = {(row["direction"], row["seed"], row["model"]): row["unit_audit_sha256"] for row in static["units"]}
            current_units, current_provenance = read_audited_units(run, protocol, configs[protocol], plans[protocol], expected_hashes)
            units.update(current_units)
            provenance.extend(current_provenance)
        report = build_transfer_tables(configs, plans, cells, units)
        for protocol, run in runs.items():
            for name, field in (("config.json", "config_sha256"), ("progress.json", "progress_sha256"), ("static_run_audit.json", "static_run_audit_sha256")):
                if stream_sha(run / name) != evidence[protocol][field]:
                    raise ValueError("formal report parent changed during aggregation")
        report.update(run_evidence=evidence, unit_evidence=provenance, main_completion=completion, main_numerical=numerical,
            weights_loaded=False, training_performed=False, predictive_weight_replay_certified_by_this_report=False,
            statistics_review="nature-statistics: independent battery units; seed SD descriptive; source folds not target replication; undefined R2 and integrity-only exclusions retained")
        save_json(output / "report.json", report)
        counts = write_tables(output, report)
        print(json.dumps({"folder": str(output), **counts, "status": "complete static-audited shared comparison; weight replay separate"}, indent=2))
    except BaseException:
        save_json(output / "failure.json", {"traceback": traceback.format_exc(), "weights_loaded": False, "training_performed": False,
            "numerical_comparison_available": False})
        raise


if __name__ == "__main__":
    main()
