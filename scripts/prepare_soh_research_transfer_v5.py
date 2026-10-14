"""Prepare two research transfer directions without fitting any target data."""

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
from battery_smetimes.research_transfer_signals import (
    FEATURE_NAMES, RESEARCH_MODE, make_transfer_plans, shared_discharge_history,
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
    if sha(registry_file) != frozen["registry_sha256"] or sha(prepared / "manifest.json") != frozen["prepared_manifest_sha256"] or sha(integrity) != frozen["integrity_sha256"]:
        raise ValueError("original frozen registry/data/integrity inputs changed")
    registered = json.loads((audit_dir / "cell_registry.json").read_text(encoding="utf-8"))["cells"]
    source_plans = json.loads((audit_dir / "splits.json").read_text(encoding="utf-8"))["plans"]
    plans = make_transfer_plans(metadata_only(registered), source_plans, mode=RESEARCH_MODE)
    registry = load_registry(registry_file)
    folder = Path("experiments/soh_research_v5/transfer_inputs") / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(folder / "preparation.log", encoding="utf-8"), logging.StreamHandler()])
    sources = [Path("scripts/prepare_soh_research_transfer_v5.py"),
        Path("battery_smetimes/research_transfer_signals.py"),
        Path("battery_smetimes/research_generalization_splits.py"),
        Path("battery_smetimes/physical_types.py"), Path("scripts/train_same_type_forecast.py")]
    hashes = {}
    for source in sources:
        target = folder / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source)] = sha(source)
    save_json(folder / "config.json", {"mode": RESEARCH_MODE, "status": "input preparation, not model training",
        "frozen_audit": str(audit_dir.resolve()), "audit_config_sha256": sha(audit_dir / "config.json"),
        "registry_sha256": sha(registry_file), "cell_registry_sha256": sha(audit_dir / "cell_registry.json"),
        "prepared_manifest_sha256": frozen["prepared_manifest_sha256"], "integrity_sha256": frozen["integrity_sha256"],
        "source_sha256": hashes, "numpy": np.__version__, "feature_names": FEATURE_NAMES,
        "origins_source_cycles": frozen["config"]["origins_source_cycles"],
        "training_seeds": frozen["config"]["training_seeds"],
        "execution_gate": "train only after the same-type main comparison/ablations complete; smoke before full run",
        "shared_sampling": "100/200/300 seconds from cached discharge-episode prefixes; MIT 30s samples linearly regridded; NASA exact 100s samples; no native whole-window statistics",
        "units": "voltage differences in V, absolute current / documented RATED Ah in C, temperature differences in degC, observed V/T at 100s",
        "source_alignment_limit": "native discharge onset definitions differ (MIT principal constant-current segment, NASA discharge episode); this cache-based representation is an approximate alignment, not raw re-extraction or chemistry invariance",
        "physical_limit": "MIT is documented LFP; NASA model/chemistry remain unknown. Distinct families retained. Failure is a transfer limit, not corrupt data. No deployment compatibility claimed.",
        "xjtu_not_in_this_shared_protocol": "native 120s charge/discharge summaries cannot be presented as 100-300s discharge V/I/T; no invented temperature or extension",
        "label_contract": frozen["config"]["target"], "target_labels": "evaluation only; no global source+target scaler, clustering or model selection",
        "target_calibration": "not part of these zero-shot plans; any future calibration needs separate target calibration/validation/test cells",
        "input_schema": "shared_discharge_cached_v1_not_native_smetimes",
        "prior_exposure": "previously inspected public development cohorts, not blind testing"})
    save_json(folder / "splits.json", {"directions": plans})
    save_json(folder / "progress.json", {"prepared_complete": False, "completed_cells": 0, "model_training_performed": False})
    written = []
    try:
        native_manifest, cells = load_data(prepared, integrity)
        native_records = {row["cell_id"]: row for row in native_manifest["cells"]}
        ids = sorted(set().union(*(set(p["source_refit"] + p["target_test"]) for p in plans)))
        for cell_id in ids:
            cell = cells[cell_id]
            cell["physical_family"] = registered[cell_id]["physical_family"]
            cycles, common = shared_discharge_history(cell, 100, registry, mode=RESEARCH_MODE)
            destination = folder / f"{cell_id}.npz"
            np.savez_compressed(destination, cycle=cycles, common_discharge_features=common,
                label_cycle=cell["label_cycle"], soh=cell["soh"])
            written.append({"cell_id": cell_id, "metadata": cell["metadata"],
                "physical_family": cell["physical_family"],
                "reference_established_source_cycle": registered[cell_id]["reference_established_source_cycle"],
                "file": str(destination.resolve()), "sha256": sha(destination),
                "native_prepared_sha256": native_records[cell_id]["sha256"],
                "integrity_excluded_input_cycles": cell["excluded_input_cycles"],
                "signal_observations": len(cycles), "feature_width": common.shape[1],
                "missing_signal_values": int(np.isnan(common).sum())})
            save_json(folder / "progress.json", {"prepared_complete": False,
                "completed_cells": len(written), "expected_cells": len(ids), "model_training_performed": False})
            logging.info("PREPARED %s observations=%s", cell_id, len(cycles))
        save_json(folder / "manifest.json", {"schema": "shared_discharge_cached_v1_not_native_smetimes",
            "cells": written, "feature_names": FEATURE_NAMES, "directions": [p["name"] for p in plans]})
        save_json(folder / "progress.json", {"prepared_complete": True,
            "completed_cells": len(written), "expected_cells": len(ids), "model_training_performed": False})
        save_json(Path("experiments/soh_research_v5/latest_transfer_inputs.json"), {"folder": str(folder.resolve())})
        print(json.dumps({"folder": str(folder.resolve()), "prepared_cells": len(written),
            "directions": [p["name"] for p in plans], "model_training_performed": False}, indent=2))
    except BaseException:
        save_json(folder / "failure.json", {"traceback": traceback.format_exc(), "completed_cells": len(written)})
        raise


if __name__ == "__main__":
    main()
