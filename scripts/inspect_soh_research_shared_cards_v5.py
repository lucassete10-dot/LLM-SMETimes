"""Fingerprint-bound, all-cell shared-card inspection; NO models or scale fits.

Consumes a prior completed raw-input audit and current immutable prepared NPZs.
Does not re-extract or rehash raw MAT files; that remains the training loader's
separate prerequisite. Cards/eligibility/future-row perturbations are evidence
only about this prepared-input representation, not forecasting accuracy.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import traceback

import numpy as np

from battery_smetimes.research_inference import verify_source_snapshot
from battery_smetimes.research_generalization_splits import metadata_only
from battery_smetimes.research_transfer_signals import make_transfer_plans, RESEARCH_MODE
from battery_smetimes.research_transfer_sequences import shared_statistics_cards
from battery_smetimes.research_transfer_baselines import transfer_prefix
from battery_smetimes.research_transfer_vi120 import SCHEMA as VI_SCHEMA, DIRECTIONS, MINIMUM_OBSERVATIONS
from scripts.prepare_same_type_forecast import sha, save_json


SOURCES = (
    "scripts/inspect_soh_research_shared_cards_v5.py", "battery_smetimes/research_transfer_sequences.py",
    "battery_smetimes/research_transfer_baselines.py", "battery_smetimes/research_transfer_signals.py",
    "battery_smetimes/research_transfer_vi120.py", "battery_smetimes/research_sequences.py",
    "battery_smetimes/research_llm.py", "battery_smetimes/research_baselines.py",
    "battery_smetimes/research_inference.py", "battery_smetimes/research_conditions_execution.py",
    "battery_smetimes/research_generalization_splits.py", "battery_smetimes/physical_types.py",
    "battery_smetimes/forecast.py", "scripts/prepare_same_type_forecast.py",
)


def inspect_prepared_cards(prepared, prior_audit, output):
    prepared, prior_audit, output = Path(prepared).resolve(), Path(prior_audit).resolve(), Path(output).resolve()
    original = verify_source_snapshot(prepared)
    audit_config = verify_source_snapshot(prior_audit)
    for name, field in (("config.json", "prepared_config_sha256"), ("manifest.json", "prepared_manifest_sha256"),
                        ("splits.json", "splits_sha256")):
        if sha(prepared / name) != audit_config[field]:
            raise ValueError("shared prepared input differs from prior complete raw audit")
    prior = json.loads((prior_audit / "report.json").read_text(encoding="utf-8"))
    if (str(prepared) != str(Path(audit_config["prepared"]).resolve()) or not prior["complete"]
            or not prior["raw_reextraction"] or (prior_audit / "failure.json").exists()
            or not json.loads((prior_audit / "progress.json").read_text(encoding="utf-8"))["complete"]):
        raise ValueError("not a complete raw input audit of these prepared inputs")
    if original["input_schema"] != VI_SCHEMA:
        raise ValueError("this all-cell inspection currently requires the strict MIT/XJTU VI protocol")
    audit_dir = Path(original["frozen_audit"])
    if (sha(audit_dir / "config.json") != original["audit_config_sha256"]
            or sha(audit_dir / "cell_registry.json") != original["cell_registry_sha256"]):
        raise ValueError("shared prepared original audit binding changed")
    registered = json.loads((audit_dir / "cell_registry.json").read_text(encoding="utf-8"))["cells"]
    registry_file = audit_dir / "source_snapshot/battery_physical_registry.json"
    if sha(registry_file) != original["registry_sha256"]:
        raise ValueError("shared prepared physical registry changed")
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    main_plans = json.loads((audit_dir / "splits.json").read_text(encoding="utf-8"))["plans"]
    plans = make_transfer_plans(metadata_only(registered), main_plans, mode=RESEARCH_MODE, directions=DIRECTIONS)
    if json.loads((prepared / "splits.json").read_text(encoding="utf-8"))["directions"] != plans:
        raise ValueError("shared prepared roles differ from unchanged whole-cell cohorts")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    required = set().union(*(set(p["source_refit"] + p["target_test"]) for p in plans))
    if (len(manifest["cells"]) != prior["verified_cells"] or len(manifest["cells"]) != len(required)
            or {r["cell_id"] for r in manifest["cells"]} != required):
        raise ValueError("shared card inspection cohort incomplete/duplicated")
    rows = []
    for index, record in enumerate(manifest["cells"]):
        key = record["cell_id"]
        original_cell = registered[key]
        if (record["metadata"] != original_cell["metadata"] or record["physical_family"] != original_cell["physical_family"]
                or record["reference_established_source_cycle"] != original_cell["reference_established_source_cycle"]
                or record["minimum_signal_observations"] != MINIMUM_OBSERVATIONS[record["metadata"]["dataset"]]
                or sha(record["file"]) != record["sha256"]):
            raise ValueError("shared-card NPZ/physical identity/eligibility binding changed")
        # NO soh/label_cycle/capacity arrays are opened as inputs.
        with np.load(record["file"], allow_pickle=False) as arrays:
            cell = {"cell_id": key, "cycle": arrays["cycle"], "common": arrays["common_discharge_features"],
                "metadata": record["metadata"], "physical_family": record["physical_family"],
                "reference_established_source_cycle": record["reference_established_source_cycle"],
                "minimum_signal_observations": record["minimum_signal_observations"],
                "input_schema": manifest["schema"], "feature_names": manifest["feature_names"]}
        cards_by_origin = {}
        for origin in original["origins_source_cycles"]:
            row = {"cell_id": key, "dataset": cell["metadata"]["dataset"], "physical_family": cell["physical_family"],
                   "origin": origin, "prepared_file_sha256": record["sha256"]}
            try:
                baseline_prefix = transfer_prefix(cell, origin)
                cards = shared_statistics_cards(cell, origin)
            except ValueError as error:
                if "insufficient" in str(error) or "not established" in str(error):
                    rows.append({**row, "eligible": False, "reason": str(error)})
                    continue
                raise
            changed = copy.deepcopy(cell)
            # All original future rows PLUS explicit appended future rows make
            # even N100 (prepared inputs stop at 100) a non-vacuous test.
            changed["common"][changed["cycle"] > origin] = np.inf
            last = max(origin, int(changed["cycle"][-1]))
            changed["cycle"] = np.r_[changed["cycle"], np.arange(last+1, last+4)]
            changed["common"] = np.r_[changed["common"], np.full((3, len(manifest["feature_names"])), np.inf)]
            changed.update(soh=object(), label_cycle=object(), reference_capacity_ah=object(), final_life=object())
            if cards != shared_statistics_cards(changed, origin) or not np.array_equal(baseline_prefix, transfer_prefix(changed, origin), equal_nan=True):
                raise ValueError("shared representation consumed future rows or labels")
            if any(name in " ".join(cards) for name in (key, cell["metadata"]["dataset"], cell["physical_family"])):
                raise ValueError("shared card text exposed dataset/cell/family identity")
            if "Temperature unavailable in this input protocol." not in " ".join(cards):
                raise ValueError("strict VI shared cards invented temperature")
            cards_by_origin[str(origin)] = cards
            rows.append({**row, "eligible": True, "cards": len(cards), "future_row_and_label_invariance": True,
                "original_future_rows_perturbed": int(np.sum(cell["cycle"] > origin)), "synthetic_future_rows_appended": 3})
        cards_file = output / "cards" / (key + ".json")
        if not cards_file.resolve().is_relative_to((output / "cards").resolve()):
            raise ValueError("unsafe cell identifier")
        cards_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(cards_file, {"cell_id": key, "dataset": cell["metadata"]["dataset"],
            "physical_family": cell["physical_family"], "cards_by_origin": cards_by_origin})
        save_json(output / "progress.json", {"complete": False, "checked_cells": index+1, "expected_cells": len(required),
            "rows": rows, "model_instantiated": False, "preprocessor_fitted": False})
        logging.info("SHARED CARD INPUT CHECK %s %s/%s", key, index+1, len(required))
    coverage = [{"dataset": domain, "origin": origin, "eligible_cells": sum(r["eligible"] for r in subset),
                 "total_cells": len(subset)} for domain in ("MIT", "XJTU") for origin in original["origins_source_cycles"]
                for subset in ([r for r in rows if r["dataset"] == domain and r["origin"] == origin],)]
    return {"complete": True, "checked_cells": len(required), "eligible_contexts": sum(r["eligible"] for r in rows),
        "coverage": coverage, "rows": rows, "model_instantiated": False, "preprocessor_fitted": False,
        "model_training_performed": False, "prediction_accuracy_verified": False,
        "raw_reextraction_performed_in_this_inspection": False, "prior_raw_audit": str(prior_audit),
        "scope": "current prepared-input fingerprints, physical identities, schema-aware cards, shared baseline-prefix eligibility and future-row/label invariance only"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    args = parser.parse_args()
    output = Path("experiments/soh_research_v5/shared_card_inspections") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "inspection.log", encoding="utf-8"), logging.StreamHandler()])
    hashes = {}
    for name in SOURCES:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
        hashes[name] = sha(destination)
    save_json(output / "config.json", {"prepared": str(args.prepared.resolve()), "input_audit": str(args.input_audit.resolve()),
        "input_audit_config_sha256": sha(args.input_audit / "config.json"),
        "input_audit_report_sha256": sha(args.input_audit / "report.json"), "source_sha256": hashes,
        "model_training_performed": False})
    try:
        report = inspect_prepared_cards(args.prepared, args.input_audit, output)
        save_json(output / "report.json", report)
        save_json(output / "progress.json", {"complete": True, "checked_cells": report["checked_cells"],
            "eligible_contexts": report["eligible_contexts"], "model_instantiated": False, "preprocessor_fitted": False})
        print(json.dumps({"folder": str(output.resolve()), **{key: value for key, value in report.items() if key != "rows"}}, indent=2))
    except BaseException:
        save_json(output / "failure.json", {"traceback": traceback.format_exc(), "model_training_performed": False})
        raise


if __name__ == "__main__":
    main()
