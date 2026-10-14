"""Write reproducible interim QA for official INL NMC/Gr 44-cell data.

Outputs are deliberately excluded from the trainable canonical v3 tables
until capacity protocol and an 80%-SOH endpoint rule are approved.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from parsers.inl_nmc_44 import parse_archive


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "inl_nmc_44"
OUT = ROOT / "data" / "interim" / "inl_nmc_44"


DIAGNOSTIC_SCHEMA = pa.schema([
    ("dataset_id", pa.string()), ("cell_id", pa.string()),
    ("pack_group", pa.string()), ("cycle_index", pa.int64()),
    ("capacity_ah", pa.float64()),
    ("coulombic_efficiency_percent", pa.float64()),
    ("end_of_charge_voltage_v", pa.float64()),
    ("end_of_discharge_voltage_v", pa.float64()),
    ("soh_first_reference_candidate", pa.float64()),
    ("soh_median5_reference_candidate", pa.float64()),
    ("soh_label_approved", pa.bool_()),
    ("rul80_cycles", pa.float64()),
])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    zip_path = RAW / "Battery raw data.zip"
    summary_path = RAW / "Pouch cell_summary.xlsx"
    cells, diagnostics = parse_archive(zip_path, summary_path)
    if len(cells) != 44 or len(diagnostics) != 24525:
        raise ValueError("unexpected official population after parsing")
    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(diagnostics, schema=DIAGNOSTIC_SCHEMA),
                   OUT / "diagnostics.parquet", compression="zstd")
    (OUT / "cells.json").write_text(
        json.dumps(cells, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    qa = {
        "dataset_id": "INL_NMC_44", "source_doi": "10.5281/zenodo.7250553",
        "source_zip_sha256": sha256(zip_path),
        "source_summary_sha256": sha256(summary_path),
        "license": "CC-BY-4.0",
        "cells": len(cells), "diagnostics": len(diagnostics),
        "unique_cell_ids": len(set(cell["cell_id"] for cell in cells)),
        "capacity_and_soh_status": "raw capacity and two reference-rule candidates; protocol not approved for supervised SOH",
        "eol80_rul_status": "not training eligible; candidate threshold crossings only",
        "first_reference_candidate_events": sum(
            cell["candidate80_first_reference_cycle"] is not None for cell in cells),
        "median5_reference_candidate_events": sum(
            cell["candidate80_median5_reference_cycle"] is not None for cell in cells),
        "reference_rule_event_cycle_disagreements": sum(
            cell["candidate80_first_reference_cycle"] !=
            cell["candidate80_median5_reference_cycle"] for cell in cells),
        "candidate_events_recovering_above_82_percent": sum(
            cell["recovered_above82pct_after_candidate"] for cell in cells),
        "cells_with_over_5_percent_capacity_jump": sum(
            cell["capacity_jumps_over_5pct_reference"] > 0 for cell in cells),
        "source_experiment_end_cycle_mismatches": sum(
            cell["last_cycle"] != cell["source_experiment_end_cycle"] for cell in cells),
        "cells_with_ce_review_flags": sum(cell["ce_review_count"] > 0 for cell in cells),
        "invalid_ce_points": sum(cell["invalid_ce_count"] for cell in cells),
        "invalid_voltage_points": sum(cell["invalid_voltage_count"] for cell in cells),
        "raw_VIT_waveforms_available": False,
        "source_channel_semantics": {
            "capacity_ah": "Amphr or CellXX in Capacity CSV, source README calls cycle-by-cycle capacity",
            "coulombic_efficiency_percent": "CE CSV values are on roughly 100% scale, not 0-1 fraction",
            "EOC_EOD_voltage_v": "end-of-charge and end-of-discharge voltage per cycle",
            "source_experiment_end_cycle": "workbook test length, not an observed 80% EOL",
        },
        "cell_level_detail": "cells.json",
        "training_eligible": False,
    }
    (OUT / "qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: qa[key] for key in (
        "cells", "diagnostics", "median5_reference_candidate_events",
        "reference_rule_event_cycle_disagreements",
        "candidate_events_recovering_above_82_percent",
        "training_eligible")}, indent=2))


if __name__ == "__main__":
    main()
