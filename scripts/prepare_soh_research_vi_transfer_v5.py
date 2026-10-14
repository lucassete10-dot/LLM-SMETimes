"""Prepare strict raw MIT/XJTU V/I transfer inputs, without model fitting."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import traceback

import numpy as np

from battery_smetimes.physical_types import load_registry
from battery_smetimes.research_generalization_splits import metadata_only
from battery_smetimes.research_transfer_signals import RESEARCH_MODE, make_transfer_plans
from battery_smetimes.research_transfer_vi120 import (
    DIRECTIONS, MAXIMUM_ENDPOINT_GAP_SECONDS, MINIMUM_OBSERVATIONS, RawVIReader, SCHEMA, VI_FEATURE_NAMES,
)
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-folder", type=Path)
    args = parser.parse_args()
    audit_dir = args.audit_folder or Path(json.loads(Path(
        "experiments/soh_research_v5/latest_audit.json").read_text(encoding="utf-8"))["folder"])
    frozen = json.loads((audit_dir / "config.json").read_text(encoding="utf-8"))
    audit = json.loads((audit_dir / "audit.json").read_text(encoding="utf-8"))
    registry_file = audit_dir / "source_snapshot/battery_physical_registry.json"
    prepared = Path(audit["prepared_folder"])
    integrity = frozen["config"]["integrity_audit"]
    if (sha(registry_file) != frozen["registry_sha256"]
            or sha(prepared / "manifest.json") != frozen["prepared_manifest_sha256"]
            or sha(integrity) != frozen["integrity_sha256"]):
        raise ValueError("original frozen registry/data/integrity inputs changed")
    registered = json.loads((audit_dir / "cell_registry.json").read_text(encoding="utf-8"))["cells"]
    source_plans = json.loads((audit_dir / "splits.json").read_text(encoding="utf-8"))["plans"]
    plans = make_transfer_plans(metadata_only(registered), source_plans,
                                mode=RESEARCH_MODE, directions=DIRECTIONS)
    registry = load_registry(registry_file)
    folder = Path("experiments/soh_research_v5/transfer_inputs") / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(folder / "preparation.log", encoding="utf-8"), logging.StreamHandler()])
    sources = [Path(name) for name in (
        "scripts/prepare_soh_research_vi_transfer_v5.py", "battery_smetimes/research_transfer_vi120.py",
        "battery_smetimes/research_transfer_signals.py", "battery_smetimes/research_generalization_splits.py",
        "battery_smetimes/physical_types.py", "scripts/train_same_type_forecast.py",
        "scripts/prepare_same_type_forecast.py", "prepare_mit.py")]
    hashes = {}
    for source in sources:
        target = folder / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source)] = sha(source)
    save_json(folder / "config.json", {
        "mode": RESEARCH_MODE, "status": "input preparation, not model training",
        "frozen_audit": str(audit_dir.resolve()), "audit_config_sha256": sha(audit_dir / "config.json"),
        "registry_sha256": sha(registry_file), "cell_registry_sha256": sha(audit_dir / "cell_registry.json"),
        "prepared_manifest_sha256": frozen["prepared_manifest_sha256"], "integrity_sha256": frozen["integrity_sha256"],
        "source_sha256": hashes, "numpy": np.__version__, "feature_names": VI_FEATURE_NAMES,
        "origins_source_cycles": frozen["config"]["origins_source_cycles"],
        "training_seeds": frozen["config"]["training_seeds"], "input_schema": SCHEMA,
        "directions": DIRECTIONS, "minimum_signal_observations": MINIMUM_OBSERVATIONS,
        "maximum_endpoint_gap_seconds": MAXIMUM_ENDPOINT_GAP_SECONDS,
        "execution_gate": "after complete same-type comparison/ablations; smoke before full transfer training",
        "shared_sampling": "raw discharge V/I at 0/30/60/90/120s; clip raw points to <=120s BEFORE interpolation; endpoint at >=114s held, never bracketed by a later point",
        "endpoint_revision": "v1 2s gap incorrectly excluded normal MIT approx5s sampling; v2 6s limit from acquisition timestamps, not target errors; initial v1 trial retained",
        "units": "voltage differences in V, absolute current / documented RATED Ah in C, observed initial V",
        "segmentation_limit": "completed OBSERVED cycle locates discharge event; MIT principal >=300s constant-current event, XJTU stage2. Not intra-cycle realtime, identical onset or chemistry invariance",
        "temperature": "omitted in BOTH datasets; never fabricated for XJTU",
        "physical_limit": "MIT LFP and XJTU NCM532 remain distinct. XJTU model identifier missing. Research transfer does not grant same-type deployment compatibility",
        "history_quality": "reuse frozen eligible native source cycles, plus strict early V/I completeness; save each new rejected cycle, retain ALL cells regardless of count or future error",
        "label_contract": frozen["config"]["target"], "target_labels": "evaluation only; no source+target scaler or target model selection",
        "target_calibration": "none; target-side calibrated adaptation requires a separate registered experiment",
        "raw_fingerprint_limit": "where original raw fingerprint was missing, record current SHA256; this is not proof of historical raw byte identity",
        "prior_exposure": "previously inspected public development cohorts, not blind tests"})
    save_json(folder / "splits.json", {"directions": plans})
    ids = sorted(set().union(*(set(p["source_refit"] + p["target_test"]) for p in plans)))
    written = []
    save_json(folder / "progress.json", {"prepared_complete": False, "completed_cells": 0,
        "expected_cells": len(ids), "model_training_performed": False})
    try:
        native_manifest, cells = load_data(prepared, integrity)
        native_records = {row["cell_id"]: row for row in native_manifest["cells"]}
        with RawVIReader(registry) as reader:
            for cell_id in ids:
                cell = cells[cell_id]
                cycles, common, rejected = reader.extract(native_records[cell_id], cell, registered[cell_id])
                destination = folder / f"{cell_id}.npz"
                np.savez_compressed(destination, cycle=cycles, common_discharge_features=common,
                    label_cycle=cell["label_cycle"], soh=cell["soh"])
                written.append({"cell_id": cell_id, "metadata": registered[cell_id]["metadata"],
                    "physical_family": registered[cell_id]["physical_family"],
                    "reference_established_source_cycle": registered[cell_id]["reference_established_source_cycle"],
                    "minimum_signal_observations": MINIMUM_OBSERVATIONS[cell["metadata"]["dataset"]],
                    "file": str(destination.resolve()), "sha256": sha(destination),
                    "native_prepared_sha256": native_records[cell_id]["sha256"],
                    "raw_source": str(Path(native_records[cell_id]["raw_source"]).resolve()),
                    "integrity_excluded_input_cycles": cell["excluded_input_cycles"],
                    "strict_prefix_rejections": rejected, "signal_observations": len(cycles),
                    "feature_width": common.shape[1], "missing_signal_values": int(np.isnan(common).sum())})
                save_json(folder / "partial_manifest.json", {"schema": SCHEMA, "cells": written,
                    "prepared_complete": False, "feature_names": VI_FEATURE_NAMES})
                save_json(folder / "raw_source_hashes.json", reader.raw_hashes)
                save_json(folder / "progress.json", {"prepared_complete": False,
                    "completed_cells": len(written), "expected_cells": len(ids), "model_training_performed": False})
                logging.info("PREPARED %s observations=%s rejected=%s", cell_id, len(cycles), len(rejected))
        save_json(folder / "manifest.json", {"schema": SCHEMA, "cells": written,
            "feature_names": VI_FEATURE_NAMES, "directions": [p["name"] for p in plans],
            "raw_source_hashes_sha256": sha(folder / "raw_source_hashes.json")})
        coverage, unavailable = [], []
        for row in written:
            with np.load(row["file"]) as arrays:
                cycles = arrays["cycle"]
            for origin in frozen["config"]["origins_source_cycles"]:
                count = int(np.sum(cycles <= origin))
                reasons = []
                if row["reference_established_source_cycle"] > origin:
                    reasons.append("reference not established")
                if count < row["minimum_signal_observations"]:
                    reasons.append("insufficient visible diagnostic observations")
                if reasons:
                    unavailable.append({"cell_id": row["cell_id"], "origin": origin,
                        "visible_observations": count, "reasons": reasons})
        for dataset in MINIMUM_OBSERVATIONS:
            cohort = [row for row in written if row["metadata"]["dataset"] == dataset]
            for origin in frozen["config"]["origins_source_cycles"]:
                missing = {r["cell_id"] for r in unavailable if r["origin"] == origin}
                coverage.append({"dataset": dataset, "origin": origin,
                    "total_cells_retained": len(cohort), "input_eligible_cells": sum(row["cell_id"] not in missing for row in cohort)})
        save_json(folder / "preparation_quality.json", {"input_coverage_not_prediction_accuracy": coverage,
            "unavailable": unavailable, "total_cells_retained": len(written),
            "strict_prefix_rejected_cycles": sum(len(row["strict_prefix_rejections"]) for row in written),
            "no_cell_dropped": True, "no_target_error_used": True,
            "model_training_performed": False})
        save_json(folder / "progress.json", {"prepared_complete": True,
            "completed_cells": len(written), "expected_cells": len(ids), "model_training_performed": False})
        save_json(Path("experiments/soh_research_v5/latest_vi_transfer_inputs.json"), {"folder": str(folder.resolve())})
        print(json.dumps({"folder": str(folder.resolve()), "prepared_cells": len(written),
            "directions": [p["name"] for p in plans], "model_training_performed": False}, indent=2))
    except BaseException:
        save_json(folder / "failure.json", {"traceback": traceback.format_exc(), "completed_cells": len(written)})
        raise


if __name__ == "__main__":
    main()
