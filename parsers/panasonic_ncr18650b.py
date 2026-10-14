"""Conservative raw HDF5 intake for Zenodo record 7896429.

This is a *test-level* parser. A test (RTP25, L206, etc.) is not assumed to
equal an aging cycle. It deliberately emits no SOH/RUL targets until the RPT
capacity step, reference rule, and EOL convention have been protocol-audited.
"""

from collections import Counter
from pathlib import Path
import re

import h5py
import numpy as np


SIGNALS = ("test_time", "voltage", "current", "temperature",
           "chargecap", "dischargecap", "stepindex")
TEMP_SENTINEL_CUTOFF = -40.0


def _invalid_counts(arrays):
    """Flag physically implausible source values; never silently delete a test."""
    voltage = arrays["voltage"]
    current = arrays["current"]
    temperature = arrays["temperature"]
    chargecap = arrays["chargecap"]
    dischargecap = arrays["dischargecap"]
    return {
        "voltage": int(np.count_nonzero(~np.isfinite(voltage) | (voltage < 0) | (voltage > 6))),
        "current": int(np.count_nonzero(~np.isfinite(current) | (np.abs(current) > 30))),
        "temperature": int(np.count_nonzero(~np.isfinite(temperature) |
                                             (temperature < TEMP_SENTINEL_CUTOFF) |
                                             (temperature > 100))),
        "capacity": int(np.count_nonzero(~np.isfinite(chargecap) |
                                          ~np.isfinite(dischargecap) |
                                          (chargecap < 0) | (dischargecap < 0) |
                                          (chargecap > 20) | (dischargecap > 20))),
    }


def _decode(value):
    return value.decode("utf-8").strip() if isinstance(value, bytes) else str(value)


def _test_groups(tests):
    """Match test IDs numerically; source group names may have leading zeros."""
    groups = {}
    for name in tests:
        if not re.fullmatch(r"test_\d+", name):
            raise ValueError(f"unexpected test group name: {name}")
        test_id = int(name[5:])
        if test_id in groups:
            raise ValueError(f"duplicate numeric test group ID: {test_id}")
        groups[test_id] = tests[name]
    return groups


def load_test(path: Path, test_id: int):
    """Return source-named arrays for one test; invalid temperatures become NaN."""
    with h5py.File(path, "r") as file:
        group = _test_groups(file["tests"])[int(test_id)]
        missing = set(SIGNALS) - set(group)
        if missing:
            raise ValueError(f"test_{test_id}: missing signals {sorted(missing)}")
        arrays = {name: group[name][:] for name in SIGNALS}
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise ValueError(f"test_{test_id}: signals have inconsistent/empty lengths")
    if np.any(np.diff(arrays["test_time"]) <= 0):
        raise ValueError(f"test_{test_id}: source time must increase strictly")
    temperature = arrays["temperature"].astype(np.float32)
    temperature[~np.isfinite(temperature) | (temperature < TEMP_SENTINEL_CUTOFF) |
                (temperature > 100)] = np.nan
    arrays["temperature"] = temperature
    return arrays


def inspect_cell(path: Path):
    """Audit file structure and source-unit ranges without producing labels."""
    path = Path(path)
    with h5py.File(path, "r") as file:
        if "info" not in file or "tests" not in file:
            raise ValueError("expected info table and tests group")
        info = file["info"][:]
        required_info = {"idcell", "idtest", "expname", "startdatetime"}
        if not info.dtype.names or not required_info.issubset(info.dtype.names):
            raise ValueError("info table is missing required fields")
        cell_ids = {_decode(value) for value in info["idcell"]}
        if len(cell_ids) != 1:
            raise ValueError("file must represent exactly one source cell")
        ids = [int(value) for value in info["idtest"]]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate test IDs in info table")
        groups = _test_groups(file["tests"])
        if set(ids) != set(groups):
            raise ValueError("info/test group mismatch")
        ordered = sorted(info, key=lambda row: float(row["startdatetime"]))
        test_rows = []
        invalid_totals = Counter()
        for row in ordered:
            test_id = int(row["idtest"])
            group = groups[test_id]
            missing = set(SIGNALS) - set(group)
            if missing:
                raise ValueError(f"test_{test_id}: missing signals {sorted(missing)}")
            lengths = {len(group[name]) for name in SIGNALS}
            if len(lengths) != 1 or 0 in lengths:
                raise ValueError(f"test_{test_id}: empty or mismatched signal lengths")
            time = group["test_time"][:]
            if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
                raise ValueError(f"test_{test_id}: invalid time order")
            voltage = group["voltage"][:]
            current = group["current"][:]
            temperature = group["temperature"][:]
            dischargecap = group["dischargecap"][:]
            chargecap = group["chargecap"][:]
            invalid = _invalid_counts({"voltage": voltage, "current": current,
                                       "temperature": temperature,
                                       "chargecap": chargecap,
                                       "dischargecap": dischargecap})
            invalid_totals.update(invalid)
            test_rows.append({
                "test_id": test_id,
                "experiment": _decode(row["expname"]),
                "startdatetime_source": float(row["startdatetime"]),
                "samples": len(time),
                "voltage_min_source": float(np.nanmin(voltage)),
                "voltage_max_source": float(np.nanmax(voltage)),
                "current_min_source": float(np.nanmin(current)),
                "current_max_source": float(np.nanmax(current)),
                "dischargecap_max_source": float(np.nanmax(dischargecap)),
                "chargecap_max_source": float(np.nanmax(chargecap)),
                "invalid_samples": invalid,
            })
    experiments = Counter(row["experiment"] for row in test_rows)
    return {
        "dataset_id": "PANASONIC_NCR18650B",
        "cell_id": next(iter(cell_ids)),
        "source_file": str(path),
        "test_count": len(test_rows),
        "samples": sum(row["samples"] for row in test_rows),
        "experiment_counts": dict(sorted(experiments.items())),
        "invalid_temperature_samples": invalid_totals["temperature"],
        "invalid_samples_by_signal": dict(invalid_totals),
        "soh_label_status": "not_derived_protocol_audit_pending",
        "rul_label_status": "not_derived_endpoint_audit_pending",
        "tests": test_rows,
    }


def rpt_1c_capacity_candidates(path: Path):
    """Extract *derived* 25 C RTP capacity candidates from two 1 C discharges.

    The author's RTP protocol repeats a full 1 C discharge to 2.5 V twice.
    In inspected HDF5 files these are contiguous step-28 runs. Both physical
    current and cutoff voltage are checked so a changed step map cannot
    silently turn another phase into a capacity label. These are RPT points,
    not aging cycles; 80%-SOH crossing between RPTs is interval-censored.
    """
    audit = inspect_cell(path)
    rows = []
    for test in audit["tests"]:
        if test["experiment"] != "RTP25":
            continue
        arrays = load_test(path, test["test_id"])
        positions = np.flatnonzero(arrays["stepindex"] == 28)
        runs = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        capacities = []
        for run in runs:
            if len(run) < 10:
                continue
            current = arrays["current"][run]
            voltage = arrays["voltage"][run]
            capacity = arrays["dischargecap"][run]
            amount = float(np.max(capacity) - np.min(capacity))
            if (-3.65 <= float(np.median(current)) <= -3.0
                    and float(voltage[-1]) <= 2.55 and amount > 0.5):
                capacities.append(amount)
        if len(capacities) != 2:
            rows.append({"test_id": test["test_id"], "status": "incomplete_rpt",
                         "qualifying_repeat_count": len(capacities)})
            continue
        median_capacity = float(np.median(capacities))
        repeat_gap = abs(capacities[0] - capacities[1]) / median_capacity
        rows.append({"test_id": test["test_id"], "status": "two_repeat_candidate",
                     "startdatetime_source": test["startdatetime_source"],
                     "capacity_repeats_source": capacities,
                     "capacity_median_source": median_capacity,
                     "repeat_relative_gap": repeat_gap,
                     "capacity_review_flag": repeat_gap > 0.05,
                     "reference_protocol": "RTP25: two 1C discharges to 2.5V",
                     "label_status": "derived_candidate_not_training_approved"})
    good = [row for row in rows if row["status"] == "two_repeat_candidate"]
    if good:
        reference = good[0]["capacity_median_source"]
        for row in good:
            row["soh_candidate"] = row["capacity_median_source"] / reference
            row["elapsed_source_days_candidate"] = (
                row["startdatetime_source"] - good[0]["startdatetime_source"])
    return {"cell_id": audit["cell_id"], "reference_rule": "first complete RTP25 median",
            "source_capacity_unit_candidate": "Ah", "training_eligible": False,
            "rul_label_status": "not_derived; RPT crossing interval-censored",
            "rpt_rows": rows}


def aging_exposure_by_test(path: Path, profile_code, nominal_capacity_ah=3.25):
    """Causal discharge-Ah/EFC at each test start, excluding RTP throughput.

    The paper defines positive current as charge, and the HDF5 test_time is
    seconds (three-week aging tests last about 1.8 million source units).
    Integrating the *negative* current in aging-profile tests gives discharged
    Ah. This is usage exposure, not a measured capacity or an EOL label.
    """
    if nominal_capacity_ah <= 0:
        raise ValueError("nominal capacity must be positive")
    exposure = {}
    cumulative_discharge_ah = 0.0
    with h5py.File(path, "r") as file:
        groups = _test_groups(file["tests"])
        info = sorted(file["info"][:], key=lambda row: float(row["startdatetime"]))
        for row in info:
            test_id = int(row["idtest"])
            exposure[test_id] = {
                "cumulative_aging_discharge_ah": cumulative_discharge_ah,
                "equivalent_full_cycles": cumulative_discharge_ah / nominal_capacity_ah,
            }
            if _decode(row["expname"]) != profile_code:
                continue
            group = groups[test_id]
            time = group["test_time"][:]
            current = group["current"][:]
            if not np.all(np.isfinite(time)) or not np.all(np.isfinite(current)):
                raise ValueError(f"test_{test_id}: nonfinite exposure signal")
            if len(time) != len(current) or len(time) < 2 or np.any(np.diff(time) <= 0):
                raise ValueError(f"test_{test_id}: invalid exposure time/current")
            discharged = float(np.trapezoid(np.maximum(-current, 0), time) / 3600)
            cumulative_discharge_ah += discharged
    return exposure


def canonical_diagnostic_candidates(path: Path, profile_code, profile_metadata,
                                    rpt_report=None):
    """Standardize *RPT diagnostic events*, keeping cycle/RUL labels absent.

    Calendar-aging time is not a cycle index. Downstream training must opt in
    only after reference/field QA and a cell-wise split, never by concatenating
    these provisional rows with the four-source v2 cycle table.
    """
    report = rpt_report if rpt_report is not None else rpt_1c_capacity_candidates(path)
    good = [row for row in report["rpt_rows"]
            if row["status"] == "two_repeat_candidate"]
    if not good:
        return []
    reference = good[0]["capacity_median_source"]
    result = []
    for diagnostic_index, row in enumerate(good, start=1):
        result.append({
            "dataset_id": "PANASONIC_NCR18650B",
            "cell_id": report["cell_id"],
            "aging_profile_id": profile_code,
            "aging_mode": profile_metadata["aging_mode"] if profile_metadata else None,
            "chemistry": "NCA",
            "form_factor": "cylindrical 18650",
            "nominal_capacity_ah": 3.25,
            "diagnostic_index": diagnostic_index,
            "cycle_index": None,
            "elapsed_source_days_candidate": row["elapsed_source_days_candidate"],
            "cumulative_aging_discharge_ah": row.get("cumulative_aging_discharge_ah"),
            "equivalent_full_cycles": row.get("equivalent_full_cycles"),
            "diagnostic_temperature_c": 25.0,
            "aging_chamber_temperature_c": (profile_metadata or {}).get("chamber_temperature_c"),
            "measured_discharge_capacity_ah_candidate": row["capacity_median_source"],
            "reference_capacity_ah_candidate": reference,
            "soh_capacity_candidate": row["soh_candidate"],
            "capacity_review_flag": row["capacity_review_flag"],
            "label_source": "RTP25 two 1C-to-2.5V discharge repeats, median",
            "label_status": "provisional_not_training_approved",
            "rul_cycles": None,
            "eol_cycle": None,
            "waveform_test_id": row["test_id"],
        })
    return result
