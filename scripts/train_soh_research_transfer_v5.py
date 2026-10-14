"""Executable zero-target-label baseline transfer, gated after full main runs.

This first executor supports Ridge/ExtraTrees, not the complete generalization
deliverable or evidence about LLM transfer. It uses a separate results root and
never writes the concurrent main summary.csv.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import time
import traceback
from contextlib import ExitStack

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "2")

import joblib
import numpy as np

from battery_smetimes.research_phase_gate import (
    assert_main_complete, assert_main_numerical_audits, main_run_bindings, select_source_candidate,
)
from battery_smetimes.research_inference import verify_source_snapshot
from battery_smetimes.research_generalization_splits import metadata_only
from battery_smetimes.research_transfer_baselines import evaluate_transfer, fit_transfer
from battery_smetimes.research_transfer_signals import RESEARCH_MODE, make_transfer_plans, shared_discharge_history
from battery_smetimes.research_transfer_vi120 import (
    DIRECTIONS as VI_DIRECTIONS, MAXIMUM_ENDPOINT_GAP_SECONDS, MINIMUM_OBSERVATIONS, RawVIReader,
    SCHEMA as VI_SCHEMA, VI_FEATURE_NAMES, stream_sha,
)
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data
from scripts.train_soh_research_v5 import save_scores


def load_shared_cells(prepared, *, progress_callback=None):
    input_config = verify_source_snapshot(prepared)
    if np.__version__ != input_config["numpy"]:
        raise ValueError("shared-input NumPy version changed")
    audit_dir = Path(input_config["frozen_audit"])
    if sha(audit_dir / "config.json") != input_config["audit_config_sha256"]:
        raise ValueError("original transfer audit changed")
    frozen = json.loads((audit_dir / "config.json").read_text(encoding="utf-8"))
    registered = json.loads((audit_dir / "cell_registry.json").read_text(encoding="utf-8"))["cells"]
    if sha(audit_dir / "cell_registry.json") != input_config["cell_registry_sha256"]:
        raise ValueError("frozen transfer cell registry changed")
    from battery_smetimes.physical_types import load_registry
    registry_file = audit_dir / "source_snapshot/battery_physical_registry.json"
    if sha(registry_file) != input_config["registry_sha256"] or sha(frozen["config"]["integrity_audit"]) != input_config["integrity_sha256"]:
        raise ValueError("transfer physical registry or integrity rules changed")
    registry = load_registry(registry_file)
    audit = json.loads((audit_dir / "audit.json").read_text(encoding="utf-8"))
    native_folder = Path(audit["prepared_folder"])
    if sha(native_folder / "manifest.json") != input_config["prepared_manifest_sha256"]:
        raise ValueError("original native prepared manifest changed")
    native_manifest, native_cells = load_data(native_folder, frozen["config"]["integrity_audit"])
    native_records = {r["cell_id"]: r for r in native_manifest["cells"]}
    protocol = input_config["input_schema"]
    if protocol not in (VI_SCHEMA, "shared_discharge_cached_v1_not_native_smetimes"):
        raise ValueError("wrong shared transfer input schema")
    directions = VI_DIRECTIONS if protocol == VI_SCHEMA else (("MIT", "NASA"), ("NASA", "MIT"))
    original_plans = json.loads((audit_dir / "splits.json").read_text(encoding="utf-8"))["plans"]
    expected_plans = make_transfer_plans(metadata_only(registered), original_plans,
        mode=RESEARCH_MODE, directions=directions)
    saved_plans = json.loads((prepared / "splits.json").read_text(encoding="utf-8"))["directions"]
    if saved_plans != expected_plans:
        raise ValueError("transfer roles differ from the metadata-only frozen plan")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != protocol or manifest["feature_names"] != input_config["feature_names"]:
        raise ValueError("wrong shared transfer input schema")
    if protocol == VI_SCHEMA:
        if (input_config["feature_names"] != VI_FEATURE_NAMES
                or input_config["minimum_signal_observations"] != MINIMUM_OBSERVATIONS
                or input_config["maximum_endpoint_gap_seconds"] != MAXIMUM_ENDPOINT_GAP_SECONDS):
            raise ValueError("strict VI protocol feature/minimum-observation contract changed")
        hash_file = prepared / "raw_source_hashes.json"
        if sha(hash_file) != manifest["raw_source_hashes_sha256"]:
            raise ValueError("raw transfer fingerprint record changed")
        raw_hashes = json.loads(hash_file.read_text(encoding="utf-8"))
        required_sources = {str(Path(native_records[key]["raw_source"]).resolve())
            for plan in expected_plans for key in plan["source_refit"] + plan["target_test"]}
        if set(raw_hashes) != required_sources:
            raise ValueError("raw transfer sources differ from frozen cohorts")
        for path, expected in raw_hashes.items():
            if stream_sha(path) != expected:
                raise ValueError("raw transfer source fingerprint changed")
    cells = {}
    with ExitStack() as stack:
        reader = stack.enter_context(RawVIReader(registry)) if protocol == VI_SCHEMA else None
        # Fingerprints were checked before any extraction; avoid hashing GB files twice.
        if reader:
            reader.raw_hashes.update(raw_hashes)
        for row in manifest["cells"]:
            key = row["cell_id"]
            if key in cells or sha(row["file"]) != row["sha256"]:
                raise ValueError("duplicate or modified transfer input")
            if row["native_prepared_sha256"] != native_records[key]["sha256"]:
                raise ValueError("shared transfer native provenance fingerprint changed")
            original = registered[key]
            if row["metadata"] != original["metadata"] or row["physical_family"] != original["physical_family"] or row["reference_established_source_cycle"] != original["reference_established_source_cycle"]:
                raise ValueError("transfer metadata/family/reference differs from original audit")
            with np.load(row["file"]) as arrays:
                cells[key] = {"cell_id": key, "metadata": row["metadata"],
                    "physical_family": row["physical_family"], "input_schema": protocol,
                    "feature_names": input_config["feature_names"],
                    "reference_established_source_cycle": row["reference_established_source_cycle"],
                    "cycle": arrays["cycle"], "common": arrays["common_discharge_features"],
                    "label_cycle": arrays["label_cycle"], "soh": arrays["soh"]}
            native = native_cells[key]
            native["physical_family"] = original["physical_family"]
            if reader:
                minimum = MINIMUM_OBSERVATIONS[row["metadata"]["dataset"]]
                if row["minimum_signal_observations"] != minimum or row["raw_source"] != str(Path(native_records[key]["raw_source"]).resolve()):
                    raise ValueError("strict VI provenance/minimum-observation metadata changed")
                cells[key]["minimum_signal_observations"] = minimum
                cycles, common, rejected = reader.extract(native_records[key], native, original)
                if rejected != row["strict_prefix_rejections"] or row["integrity_excluded_input_cycles"] != native["excluded_input_cycles"]:
                    raise ValueError("strict VI exclusion reasons differ from raw audited history")
            else:
                cycles, common = shared_discharge_history(native, 100, registry, mode=RESEARCH_MODE)
            for name, expected in (("cycle", cycles), ("common", common),
                                   ("label_cycle", native["label_cycle"]), ("soh", native["soh"])):
                if not np.array_equal(cells[key][name], expected, equal_nan=True):
                    raise ValueError("derived shared signals/targets disagree with original audited data")
            if progress_callback:
                progress_callback(key, len(cells), len(manifest["cells"]))
    required = set().union(*(set(plan["source_refit"] + plan["target_test"]) for plan in expected_plans))
    if set(cells) != required:
        raise ValueError("shared transfer inputs omit or add a frozen cohort cell")
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--protocol", choices=("cached_vit", "strict_vi120"), default="cached_vit")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    pointer = "latest_vi_transfer_inputs.json" if args.protocol == "strict_vi120" else "latest_transfer_inputs.json"
    prepared = args.prepared or Path(json.loads((Path(
        "experiments/soh_research_v5") / pointer).read_text(encoding="utf-8"))["folder"])
    input_config = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    if not json.loads((prepared / "progress.json").read_text(encoding="utf-8"))["prepared_complete"]:
        raise ValueError("transfer input preparation is incomplete")
    audit = Path(input_config["frozen_audit"])
    if sha(audit / "config.json") != input_config["audit_config_sha256"]:
        raise ValueError("transfer's original frozen audit changed")
    frozen = json.loads((audit / "config.json").read_text(encoding="utf-8"))
    candidates = frozen["config"]["initial_candidates"]
    plans = json.loads((prepared / "splits.json").read_text(encoding="utf-8"))["directions"]
    bindings = main_run_bindings("experiments/soh_research_v5/runs", input_config["audit_config_sha256"])
    folder = Path("experiments/soh_research_v5/transfer_runs") / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(folder / "training.log", encoding="utf-8"), logging.StreamHandler()])
    sources = [Path("scripts/train_soh_research_transfer_v5.py"),
        Path("battery_smetimes/research_transfer_baselines.py"),
        Path("battery_smetimes/research_phase_gate.py"), Path("battery_smetimes/research_inference.py"),
        Path("battery_smetimes/research_transfer_signals.py"), Path("battery_smetimes/research_baselines.py"),
        Path("battery_smetimes/research_transfer_vi120.py"), Path("prepare_mit.py"),
        Path("battery_smetimes/research_generalization_splits.py"),
        Path("battery_smetimes/physical_types.py"), Path("scripts/train_same_type_forecast.py"),
        Path("battery_smetimes/forecast.py"), Path("scripts/train_soh_research_v5.py"),
        Path("scripts/prepare_same_type_forecast.py")]
    hashes = {}
    for source in sources:
        target = folder / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source)] = sha(source)
    seeds = input_config["training_seeds"][:1] if args.smoke else input_config["training_seeds"]
    origins = input_config["origins_source_cycles"]
    save_json(folder / "config.json", {"prepared": str(prepared.resolve()),
        "prepared_manifest_sha256": sha(prepared / "manifest.json"), "input_config_sha256": sha(prepared / "config.json"),
        "splits_sha256": sha(prepared / "splits.json"), "source_sha256": hashes,
        "main_run_bindings": bindings, "smoke": args.smoke, "seeds": seeds, "origins": origins,
        "candidate_budget": candidates, "numpy": np.__version__,
        "source_selection_folds": "first single fold for smoke; all source folds for full",
        "model_scope": "shared-input Ridge/ExtraTrees baseline transfer only; no LLM contribution claim",
        "evidence_level": "engineering smoke" if args.smoke else "public development cross-family research stress test"})
    save_json(folder / "splits.json", {"directions": plans})
    state = {"complete": False, "completed": [], "model_training_started": False,
             "expected_final_model_units": len(plans) * len(seeds) * len(candidates)}
    save_json(folder / "progress.json", state)
    try:
        gate = assert_main_complete(bindings)
        save_json(folder / "main_completion_gate.json", {"passed": True, "evidence": gate})
        numerical = assert_main_numerical_audits(bindings,
            "experiments/soh_research_v5/artifact_audits", completion=gate)
        save_json(folder / "main_numerical_audit_gate.json", {"passed": True, "evidence": numerical})
        cells = load_shared_cells(prepared)
        for plan in plans:
            folds = plan["source_selection_folds"][:1] if args.smoke else plan["source_selection_folds"]
            required = [key for fold in folds for key in fold["source_validation"]]
            for seed in seeds:
                for model_name, settings in candidates.items():
                    unit = folder / plan["name"] / str(seed) / model_name
                    unit.mkdir(parents=True, exist_ok=False)
                    save_json(unit / "config.json", {"plan": plan, "seed": seed, "model": model_name,
                        "parameters": settings, "origins": origins, "source_selection_folds": folds})
                    selection_rows = []
                    started = time.perf_counter()
                    state["model_training_started"] = True
                    save_json(folder / "progress.json", state)
                    for index, parameters in enumerate(settings):
                        validation = []
                        for fold in folds:
                            destination = unit / f"candidate_{index}" / fold["name"]
                            destination.mkdir(parents=True, exist_ok=False)
                            logging.info("SOURCE FIT %s %s seed=%s candidate=%s fold=%s", plan["name"], model_name, seed, index, fold["name"])
                            bundle = fit_transfer(cells, fold["source_train"], plan, origins,
                                model_name, parameters, seed, args.smoke)
                            joblib.dump(bundle, destination / "model.joblib")
                            report, arrays = evaluate_transfer(bundle, cells, fold["source_validation"])
                            save_scores(destination / "source_validation", report, arrays)
                            validation.extend(report["by_cell"])
                            save_json(destination / "resources.json", {"fit_seconds": bundle["fit_seconds"],
                                "source_only_preprocessor_fit_ids": bundle["source_only_preprocessor_fit_ids"]})
                        selection_rows.append({"index": index, "parameters": parameters, "by_cell": validation})
                        save_json(unit / "source_validation_candidates.json", selection_rows)
                    selected, scores = select_source_candidate(selection_rows, required)
                    save_json(unit / "selected.json", selected)
                    save_json(unit / "source_validation_scores.json", scores)
                    bundle = fit_transfer(cells, plan["source_refit"], plan, origins,
                        model_name, selected["parameters"], seed, args.smoke)
                    joblib.dump(bundle, unit / "model.joblib")
                    report, arrays = evaluate_transfer(bundle, cells, plan["target_test"])
                    save_scores(unit / "target_test", report, arrays)
                    save_json(unit / "resources.json", {"fit_and_evaluate_seconds": time.perf_counter() - started,
                        "source_only_preprocessor_fit_ids": bundle["source_only_preprocessor_fit_ids"],
                        "target_calibration": [], "tree_estimators": 48 if args.smoke else 160,
                        "cpu_measurement_under_background_load_not_latency_benchmark": True})
                    state["completed"].append({"direction": plan["name"], "seed": seed, "model": model_name})
                    save_json(folder / "progress.json", state)
                    logging.info("SAVED TARGET EVALUATION %s %s seed=%s", plan["name"], model_name, seed)
        state["complete"] = True
        save_json(folder / "progress.json", state)
        print(json.dumps({"folder": str(folder.resolve()), "completed_units": len(state["completed"]),
                          "smoke": args.smoke}, indent=2))
    except BaseException:
        save_json(folder / "failure.json", {"traceback": traceback.format_exc(),
            "model_training_started": state["model_training_started"], "completed": state["completed"]})
        raise


if __name__ == "__main__":
    main()
