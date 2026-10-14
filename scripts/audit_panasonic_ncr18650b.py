"""Audit downloaded Panasonic cells without adding them to the training set."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parsers.panasonic_ncr18650b import (  # noqa: E402
    aging_exposure_by_test, canonical_diagnostic_candidates, inspect_cell,
    rpt_1c_capacity_candidates,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_interval(rpt_rows, threshold=0.8):
    """Bound the first sustained crossing between two sparse RTP checks."""
    rows = [row for row in rpt_rows if row["status"] == "two_repeat_candidate"]
    for index in range(1, len(rows) - 1):
        if (rows[index - 1]["soh_candidate"] > threshold
                and rows[index]["soh_candidate"] <= threshold
                and rows[index + 1]["soh_candidate"] <= threshold):
            return {
                "status": "interval_observed_not_exact_cycle",
                "threshold": threshold,
                "last_above_test_id": rows[index - 1]["test_id"],
                "first_below_test_id": rows[index]["test_id"],
                "last_above_elapsed_source_days_candidate":
                    rows[index - 1]["elapsed_source_days_candidate"],
                "first_below_elapsed_source_days_candidate":
                    rows[index]["elapsed_source_days_candidate"],
                "last_above_equivalent_full_cycles_candidate":
                    rows[index - 1].get("equivalent_full_cycles"),
                "first_below_equivalent_full_cycles_candidate":
                    rows[index].get("equivalent_full_cycles"),
            }
    return {"status": "not_observed_or_not_confirmed", "threshold": threshold}


def audit(raw_dir, manifest, protocols):
    source_hashes = {row.get("measurement_sha256") for row in manifest["cells"]}
    source_hashes.discard(None)
    cells = []
    diagnostics = []
    for path in sorted(Path(raw_dir).glob("2_*.h5")):
        cell = inspect_cell(path)
        profile_codes = [code for code in cell["experiment_counts"]
                         if code in protocols["profiles"]]
        if len(profile_codes) == 1:
            cell["aging_profile_code"] = profile_codes[0]
            cell["aging_profile_metadata"] = protocols["profiles"][profile_codes[0]]
        else:
            cell["aging_profile_code"] = None
            cell["aging_profile_metadata"] = None
        cell["rpt_capacity_candidates"] = rpt_1c_capacity_candidates(path)
        if cell["aging_profile_code"]:
            exposure = aging_exposure_by_test(path, cell["aging_profile_code"])
            for row in cell["rpt_capacity_candidates"]["rpt_rows"]:
                if row["status"] == "two_repeat_candidate":
                    row.update(exposure[row["test_id"]])
        valid_rpt = sum(row["status"] == "two_repeat_candidate"
                        for row in cell["rpt_capacity_candidates"]["rpt_rows"])
        cell["qualified_rpt_count"] = valid_rpt
        cell["rpt_capacity_review_count"] = sum(
            bool(row.get("capacity_review_flag"))
            for row in cell["rpt_capacity_candidates"]["rpt_rows"])
        cell["eol80_rpt_interval"] = threshold_interval(
            cell["rpt_capacity_candidates"]["rpt_rows"])
        cell["longitudinal_soh_candidate_available"] = valid_rpt >= 2
        cell["cycle_rul_label_available"] = False
        diagnostics.extend(canonical_diagnostic_candidates(
            path, cell["aging_profile_code"], cell["aging_profile_metadata"],
            cell["rpt_capacity_candidates"]))
        checksum = sha256(path)
        cell["sha256"] = checksum
        cell["duplicate_source_hash_in_v2"] = checksum in source_hashes
        cells.append(cell)
    ids = [cell["cell_id"] for cell in cells]
    checksums = [cell["sha256"] for cell in cells]
    report = {
        "dataset_id": "PANASONIC_NCR18650B",
        "source_record": "https://zenodo.org/records/7896429",
        "training_eligible": False,
        "reason": "RPT capacity reference, aging-cycle alignment, and 80%-SOH endpoint not audited",
        "cell_count_downloaded": len(cells),
        "duplicate_cell_ids_within_download": len(ids) != len(set(ids)),
        "duplicate_file_sha256_within_download": len(checksums) != len(set(checksums)),
        "source_hash_comparison_scope": "Only v2 records with measurement_sha256; MIT and XJTU have no such field",
        "cells": cells,
    }
    return report, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/panasonic_ncr18650b")
    parser.add_argument("--manifest", type=Path, default=ROOT / "battery_data_v2/manifest.json")
    parser.add_argument("--protocols", type=Path,
                        default=ROOT / "metadata/panasonic_protocols.yaml")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/interim/panasonic_ncr18650b/intake_qa.json")
    parser.add_argument("--diagnostics-output", type=Path,
                        default=ROOT / "data/interim/panasonic_ncr18650b/diagnostic_candidates.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocols = yaml.safe_load(args.protocols.read_text(encoding="utf-8"))
    result, diagnostics = audit(args.raw_dir, manifest, protocols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_output.write_text(json.dumps({
        "schema_version": 1,
        "training_eligible": False,
        "reason": "Provisional RPT SOH only; 80%-SOH EOL interval and cyclic exposure not established",
        "rows": diagnostics,
    }, indent=2), encoding="utf-8")
    print(f"Audited {result['cell_count_downloaded']} downloaded cells: {args.output}")
    print(f"Wrote {len(diagnostics)} provisional RPT rows: {args.diagnostics_output}")


if __name__ == "__main__":
    main()
