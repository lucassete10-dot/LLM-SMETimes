"""Materialize auditable cross-dataset capacity and endpoint tables.

Run from the repository root. Source RUL and confirmed own-reference 80%-SOH
RUL are deliberately distinct targets. Feature vectors retain their source
schema; they are not asserted to be aligned across datasets.
"""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from battery_v2_features import load_cell


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
MANIFEST = ROOT / "battery_data_v2" / "manifest.json"
AUDIT = ROOT / "battery_models_v2" / "exact80_target_audit.json"


CELL_SCHEMA = pa.schema([
    ("dataset_id", pa.string()), ("cell_id", pa.string()),
    ("group_id", pa.string()), ("cycle_unit", pa.string()),
    ("chemistry", pa.string()), ("form_factor", pa.string()),
    ("nominal_capacity_ah", pa.float64()),
    ("reference_capacity_ah", pa.float64()),
    ("reference_rule", pa.string()),
    ("reference_established_at_cycle", pa.int64()),
    ("reference_measurement_count", pa.int32()),
    ("source_endpoint_rule", pa.string()),
    ("source_endpoint_cycle", pa.int64()),
    ("source_endpoint_observed", pa.bool_()),
    ("source_censor_cycle", pa.int64()),
    ("exact80_threshold_ah", pa.float64()),
    ("exact80_first_observed_crossing_cycle", pa.int64()),
    ("exact80_confirmed_event_cycle", pa.int64()),
    ("exact80_confirmed_event_observed", pa.bool_()),
    ("last_observed_soh_cycle", pa.int64()),
    ("observed_through_cycle", pa.int64()),
    ("charge_protocol", pa.string()),
    ("charge_rate_c", pa.float64()),
    ("discharge_rate_c", pa.float64()),
    ("diagnostic_discharge_rate_c", pa.float64()),
    ("ambient_temperature_c", pa.float64()),
    ("temperature_context", pa.string()),
    ("measurement_source", pa.string()),
    ("measurement_sha256", pa.string()),
    ("feature_source", pa.string()),
    ("feature_sha256", pa.string()),
    ("labels_file", pa.string()),
])

CYCLE_SCHEMA = pa.schema([
    ("dataset_id", pa.string()), ("cell_id", pa.string()),
    ("group_id", pa.string()), ("cycle_index", pa.int64()),
    ("source_cycle", pa.int64()), ("cycle_unit", pa.string()),
    ("timestamp_s", pa.float64()),
    ("measured_discharge_capacity_ah", pa.float64()),
    ("measured_charge_capacity_ah", pa.float64()),
    ("soh_capacity", pa.float64()), ("soh_available", pa.bool_()),
    ("soh_label_source", pa.string()),
    ("reference_calibration_row", pa.bool_()),
    ("source_rul_cycles", pa.float64()),
    ("source_rul_available", pa.bool_()),
    ("exact80_rul_cycles", pa.float64()),
    ("exact80_rul_available", pa.bool_()),
    ("exact80_rul_lower_bound_cycles", pa.float64()),
    ("capacity_review_flag", pa.bool_()),
    ("ambient_temperature_c", pa.float64()),
    ("charge_temperature_mean_c", pa.float64()),
    ("discharge_temperature_mean_c", pa.float64()),
    ("cumulative_ah_throughput", pa.float64()),
    ("cumulative_wh_throughput", pa.float64()),
    ("dcir_ohm", pa.float64()),
    ("quality_note", pa.string()),
])

FEATURE_SCHEMA = pa.schema([
    ("dataset_id", pa.string()), ("cell_id", pa.string()),
    ("cycle_index", pa.int64()), ("feature_schema_id", pa.string()),
    ("raw_signal_features", pa.list_(pa.float32())),
    ("feature_source", pa.string()),
])


def audit_index(audit):
    return {(dataset, row["cell_id"]): row
            for dataset, domain in audit["domains"].items()
            for row in domain["by_cell"]}


def cell_row(record, event):
    if not math.isclose(record["reference_capacity_ah"],
                        event["reference_capacity_ah"], rel_tol=1e-6):
        raise ValueError(f"reference mismatch: {record['cell_id']}")
    event_cycle = event["first_confirmed_exact80_cycle"]
    observed = event["last_observed_soh_cycle"]
    if event_cycle is not None and (observed is None or event_cycle > observed):
        raise ValueError(f"event after last observation: {record['cell_id']}")
    if record["endpoint_observed"] != (record["endpoint_cycle"] is not None):
        raise ValueError(f"source endpoint mask mismatch: {record['cell_id']}")
    return {
        "dataset_id": record["dataset"], "cell_id": record["cell_id"],
        "group_id": record["group"], "cycle_unit": record["cycle_unit"],
        "chemistry": record.get("chemistry"), "form_factor": None,
        "nominal_capacity_ah": None,
        "reference_capacity_ah": record["reference_capacity_ah"],
        "reference_rule": record["reference_rule"],
        "reference_established_at_cycle": record["reference_established_at_cycle"],
        "reference_measurement_count": record["reference_measurement_count"],
        "source_endpoint_rule": record["endpoint_rule"],
        "source_endpoint_cycle": record["endpoint_cycle"],
        "source_endpoint_observed": record["endpoint_observed"],
        "source_censor_cycle": record.get("censor_cycle"),
        "exact80_threshold_ah": event["exact80_threshold_ah"],
        "exact80_first_observed_crossing_cycle": event["first_observed_exact80_cycle"],
        "exact80_confirmed_event_cycle": event_cycle,
        "exact80_confirmed_event_observed": event_cycle is not None,
        "last_observed_soh_cycle": observed,
        "observed_through_cycle": record["observed_through_cycle"],
        "charge_protocol": record.get("charge_protocol"),
        "charge_rate_c": record.get("charge_rate_c"),
        "discharge_rate_c": record.get("discharge_rate_c"),
        "diagnostic_discharge_rate_c": record.get("diagnostic_discharge_rate_c"),
        "ambient_temperature_c": record.get("ambient_temperature_c"),
        "temperature_context": record.get("temperature_context"),
        "measurement_source": record.get("measurement_source"),
        "measurement_sha256": record.get("measurement_sha256"),
        "feature_source": record.get("feature_source"),
        "feature_sha256": record.get("feature_sha256"),
        "labels_file": record["labels_file"],
    }


def cycle_rows(record, event, labels):
    cycle = labels["cycle"].astype("int64")
    source_cycle = labels["source_cycle"].astype("int64")
    capacity = labels["capacity_ah"].astype("float64")
    soh = labels["soh"].astype("float64")
    soh_available = labels["soh_available"].astype(bool)
    source_rul = labels["rul_cycles"].astype("float64")
    source_available = labels["rul_available"].astype(bool)
    review = labels["capacity_review_flag"].astype(bool)
    n = len(cycle)
    if any(len(x) != n for x in (source_cycle, capacity, soh,
                                  soh_available, source_rul,
                                  source_available, review)):
        raise ValueError(f"array length mismatch: {record['cell_id']}")
    if n != record["rows"] or np.any(np.diff(cycle) <= 0):
        raise ValueError(f"cycle count/order mismatch: {record['cell_id']}")
    if not np.all(np.isfinite(capacity)) or np.any(capacity <= 0):
        raise ValueError(f"invalid capacity: {record['cell_id']}")
    expected = capacity / record["reference_capacity_ah"]
    if not np.allclose(soh[soh_available], expected[soh_available],
                       rtol=1e-5, atol=1e-5):
        raise ValueError(f"SOH/reference mismatch: {record['cell_id']}")
    if np.any(np.isfinite(soh[~soh_available])) or np.any(np.isfinite(source_rul[~source_available])):
        raise ValueError(f"unmasked label: {record['cell_id']}")
    endpoint = event["first_confirmed_exact80_cycle"]
    last = event["last_observed_soh_cycle"]
    rows = []
    for i, c in enumerate(cycle):
        c = int(c)
        eligible = bool(soh_available[i])
        event_rul = (float(endpoint - c) if eligible and endpoint is not None
                     and c <= endpoint else None)
        lower_bound = (float(last - c) if eligible and endpoint is None
                       and last is not None and c <= last else None)
        rows.append({
            "dataset_id": record["dataset"], "cell_id": record["cell_id"],
            "group_id": record["group"], "cycle_index": c,
            "source_cycle": int(source_cycle[i]), "cycle_unit": record["cycle_unit"],
            "timestamp_s": None,
            "measured_discharge_capacity_ah": float(capacity[i]),
            "measured_charge_capacity_ah": None,
            "soh_capacity": float(soh[i]) if eligible else None,
            "soh_available": eligible,
            "soh_label_source": ("measured discharge capacity / per-cell reference"
                                 if eligible else None),
            "reference_calibration_row": not eligible,
            "source_rul_cycles": float(source_rul[i]) if source_available[i] else None,
            "source_rul_available": bool(source_available[i]),
            "exact80_rul_cycles": event_rul,
            "exact80_rul_available": event_rul is not None,
            "exact80_rul_lower_bound_cycles": lower_bound,
            "capacity_review_flag": bool(review[i]),
            "ambient_temperature_c": record.get("ambient_temperature_c"),
            "charge_temperature_mean_c": None,
            "discharge_temperature_mean_c": None,
            "cumulative_ah_throughput": None,
            "cumulative_wh_throughput": None,
            "dcir_ohm": None,
            "quality_note": "capacity local-median review" if review[i] else None,
        })
    return rows


def build():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from repository root: {ROOT}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    records = manifest["cells"]
    events = audit_index(audit)
    keys = [(r["dataset"], r["cell_id"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != set(events):
        raise ValueError("duplicate or missing dataset/cell audit keys")
    OUT.mkdir(parents=True, exist_ok=True)
    cell_rows = [cell_row(record, events[key]) for record, key in zip(records, keys)]
    stats = {d: {"cells": 0, "cycles": 0, "confirmed_exact80_events": 0,
                 "exact80_point_labels": 0, "exact80_censored_cells": 0}
             for d in manifest["summary"]}
    feature_schemas = {}
    cycle_path = OUT / "cycles.parquet"
    feature_path = OUT / "features.parquet"
    with pq.ParquetWriter(cycle_path, CYCLE_SCHEMA, compression="zstd") as cycle_writer, \
         pq.ParquetWriter(feature_path, FEATURE_SCHEMA, compression="zstd") as feature_writer:
        for record, key in zip(records, keys):
            event = events[key]
            with np.load(ROOT / record["labels_file"]) as labels:
                rows = cycle_rows(record, event, labels)
            cycle_writer.write_table(pa.Table.from_pylist(rows, schema=CYCLE_SCHEMA))
            cell = load_cell(record)
            if not np.array_equal(cell.cycle, [r["cycle_index"] for r in rows]):
                raise ValueError(f"feature/label cycle mismatch: {key}")
            schema_id = f"{record['dataset']}:{record.get('feature_array', 'nasa_1000s')}:v2:{cell.raw.shape[1]}"
            previous = feature_schemas.get(schema_id)
            schema_info = {"dataset_id": record["dataset"],
                           "feature_array": record.get("feature_array", "nasa_1000s"),
                           "width": int(cell.raw.shape[1]),
                           "alignment_status": "source-specific; not cross-dataset aligned",
                           "capacity_or_soh_as_input": False}
            if previous is not None and previous != schema_info:
                raise ValueError(f"inconsistent feature schema: {schema_id}")
            feature_schemas[schema_id] = schema_info
            feature_rows = [{"dataset_id": record["dataset"],
                             "cell_id": record["cell_id"], "cycle_index": int(c),
                             "feature_schema_id": schema_id,
                             "raw_signal_features": raw.tolist(),
                             "feature_source": record.get("feature_source") or record["measurement_source"]}
                            for c, raw in zip(cell.cycle, cell.raw)]
            feature_writer.write_table(pa.Table.from_pylist(feature_rows, schema=FEATURE_SCHEMA))
            domain = stats[record["dataset"]]
            domain["cells"] += 1
            domain["cycles"] += len(rows)
            domain["confirmed_exact80_events"] += int(event["first_confirmed_exact80_cycle"] is not None)
            domain["exact80_censored_cells"] += int(event["first_confirmed_exact80_cycle"] is None)
            domain["exact80_point_labels"] += sum(r["exact80_rul_available"] for r in rows)
    pq.write_table(pa.Table.from_pylist(cell_rows, schema=CELL_SCHEMA),
                   OUT / "cells.parquet", compression="zstd")
    if any(stats[d]["cells"] != s["cells"] or
           stats[d]["cycles"] != s["observations"] for d, s in manifest["summary"].items()):
        raise ValueError("output coverage does not match v2 manifest")
    (OUT / "feature_schemas.json").write_text(
        json.dumps(feature_schemas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    build_manifest = {
        "schema_version": 3, "source_labels_manifest": str(MANIFEST.relative_to(ROOT)),
        "exact80_audit": str(AUDIT.relative_to(ROOT)),
        "exact80_event_definition": audit["confirmed_event"],
        "point_rul_rule": "confirmed event cycle minus current cycle; no label after event",
        "censor_rule": "no exact80 point RUL; lower bound through last observed SOH cycle",
        "feature_alignment": "source-specific raw signal vectors; do not concatenate without mapping",
        "unavailable_physical_fields": "null, never imputed in canonical source tables",
        "stats": stats,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(build_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    build()
