"""Independently rederive every prepared transfer input; never fit a model."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import traceback

from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_soh_research_transfer_v5 import load_shared_cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    prepared = args.prepared.resolve()
    folder = Path("experiments/soh_research_v5/transfer_input_audits") / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(folder / "verification.log", encoding="utf-8"), logging.StreamHandler()])
    sources = [Path(name) for name in (
        "scripts/verify_soh_research_transfer_inputs_v5.py", "scripts/train_soh_research_transfer_v5.py",
        "battery_smetimes/research_transfer_vi120.py", "battery_smetimes/research_transfer_signals.py",
        "battery_smetimes/research_generalization_splits.py", "battery_smetimes/research_inference.py",
        "battery_smetimes/physical_types.py", "scripts/train_same_type_forecast.py",
        "scripts/prepare_same_type_forecast.py", "prepare_mit.py")]
    hashes = {}
    for source in sources:
        destination = folder / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(source)] = sha(source)
    save_json(folder / "config.json", {"prepared": str(prepared), "source_sha256": hashes,
        "prepared_config_sha256": sha(prepared / "config.json"),
        "prepared_manifest_sha256": sha(prepared / "manifest.json"),
        "splits_sha256": sha(prepared / "splits.json"),
        "scope": "all-cell raw/cached signal, labels, provenance, quality exclusions and role verification; not model accuracy",
        "model_training_performed": False})
    state = {"complete": False, "verified_cells": 0, "model_training_performed": False}
    save_json(folder / "progress.json", state)

    def progress(key, count, expected):
        state.update(verified_cells=count, expected_cells=expected, last_cell=key)
        save_json(folder / "progress.json", state)
        logging.info("VERIFIED %s %s/%s", key, count, expected)

    try:
        if not json.loads((prepared / "progress.json").read_text(encoding="utf-8"))["prepared_complete"]:
            raise ValueError("transfer input preparation is incomplete")
        cells = load_shared_cells(prepared, progress_callback=progress)
        report = {"verified_cells": len(cells), "complete": True,
            "physical_families": sorted({cell["physical_family"] for cell in cells.values()}),
            "datasets": sorted({cell["metadata"]["dataset"] for cell in cells.values()}),
            "input_schemas": sorted({cell["input_schema"] for cell in cells.values()}),
            "raw_reextraction": "strict_raw_discharge_vi_120s_v2" in {cell["input_schema"] for cell in cells.values()},
            "model_training_performed": False, "prediction_accuracy_verified": False}
        save_json(folder / "report.json", report)
        state["complete"] = True
        save_json(folder / "progress.json", state)
        save_json(Path("experiments/soh_research_v5/latest_transfer_input_audit.json"), {"folder": str(folder.resolve())})
        print(json.dumps({"folder": str(folder.resolve()), **report}, indent=2))
    except BaseException:
        save_json(folder / "failure.json", {"traceback": traceback.format_exc(), "verified_cells": state["verified_cells"]})
        raise


if __name__ == "__main__":
    main()
