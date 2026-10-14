"""Safe read-only intake of the official INL 44-cell NMC/graphite archive.

The source contains per-cycle capacity, CE, EOCV and EODV, but no sampled
V/I/T waveforms. Reference choice and EOL event are *candidates* pending a
protocol/RPT review; this parser never emits trainable RUL point labels.
"""

from collections import defaultdict
import io
from pathlib import Path
import re
import zipfile

import numpy as np
import openpyxl
import pandas as pd


MEMBER = re.compile(
    r"Battery raw data/(P\d+)_NMC(532|811)_R([12]) (?:design|deisgn)/"
    r"(Capacity|CE|EOC|EOD)_Cell(\d+)\.csv$"
)
CHANNELS = {"Capacity", "CE", "EOC", "EOD"}


def metadata_index(workbook_path):
    workbook = openpyxl.load_workbook(workbook_path, read_only=True,
                                      data_only=True)
    try:
        result = {}
        for sheet in workbook.worksheets:
            for row in list(sheet.values)[2:]:
                if not isinstance(row[1], (int, float)):
                    continue
                key = (sheet.title, int(row[1]))
                if key in result:
                    raise ValueError(f"duplicate summary cell: {key}")
                rate = str(row[2]).strip()
                match = re.fullmatch(r"(\d+(?:\.\d+)?)C", rate)
                if not match:
                    raise ValueError(f"unparsed summary charge rate: {key}: {rate}")
                result[key] = {
                    "charge_rate_c": float(match.group(1)),
                    "charge_protocol": str(row[3]).strip(),
                    "source_experiment_end_cycle": int(row[4]),
                    "charge_voltage_limit_v": float(row[5]),
                    "source_aging_mode": str(row[6]).strip() if row[6] else None,
                    "source_training_suggestion": str(row[7]).strip() if row[7] else None,
                }
        return result
    finally:
        workbook.close()


def archive_index(archive):
    by_cell = defaultdict(dict)
    for item in archive.infolist():
        if item.is_dir():
            continue
        match = MEMBER.fullmatch(item.filename)
        if not match:
            raise ValueError(f"unexpected archive member: {item.filename}")
        pack, chemistry, design, channel, number = match.groups()
        key = (pack, int(number))
        if channel in by_cell[key]:
            raise ValueError(f"duplicate {channel} series: {key}")
        by_cell[key][channel] = {
            "member": item.filename, "chemistry": f"NMC{chemistry}/graphite",
            "electrode_design": f"R{design}",
        }
    for key, channels in by_cell.items():
        if set(channels) != CHANNELS:
            raise ValueError(f"missing channel for {key}: {CHANNELS - set(channels)}")
        if len({v["chemistry"] for v in channels.values()}) != 1 or \
           len({v["electrode_design"] for v in channels.values()}) != 1:
            raise ValueError(f"channel metadata mismatch: {key}")
    return dict(by_cell)


def read_channel(archive, member, channel, cell_number):
    frame = pd.read_csv(io.BytesIO(archive.read(member)))
    if len(frame.columns) != 3 or len(frame) == 0:
        raise ValueError(f"invalid {channel} shape: {member}")
    cycle = pd.to_numeric(frame.iloc[:, 1], errors="raise").to_numpy()
    values = pd.to_numeric(frame.iloc[:, 2], errors="raise").to_numpy(float)
    if not np.isfinite(cycle).all() or not np.array_equal(cycle, cycle.astype("int64")):
        raise ValueError(f"noninteger cycle: {member}")
    cycle = cycle.astype("int64")
    if np.any(np.diff(cycle) <= 0) or cycle[0] < 1 or not np.isfinite(values).all():
        raise ValueError(f"nonmonotone cycle or nonfinite {channel}: {member}")
    expected_value_column = (
        "CE" if channel == "CE" else
        "Amphr" if channel == "Capacity" and frame.columns[1] == "CycleReorder" else
        "Volts" if channel in ("EOC", "EOD") and
        frame.columns[1] == "CycleReorder" else
        f"Cell{cell_number:02d}"
    )
    if frame.columns[2] != expected_value_column:
        raise ValueError(f"unexpected {channel} units/column: {member}")
    return cycle, values


def first_confirmed_candidate(cycle, soh):
    """Three below 80%, plus local median; not a validated physical EOL."""
    for index in range(len(cycle) - 2):
        if (np.all(soh[index:index + 3] <= .8) and
                np.median(soh[index:min(len(soh), index + 10)]) <= .8):
            return int(cycle[index])
    return None


def parse_archive(archive_path, workbook_path):
    summary = metadata_index(workbook_path)
    diagnostics, cells = [], []
    with zipfile.ZipFile(archive_path) as archive:
        indexed = archive_index(archive)
        if set(indexed) != set(summary):
            raise ValueError("raw capacity cells and summary workbook do not match")
        for key in sorted(indexed):
            pack, number = key
            series = {}
            for channel in sorted(CHANNELS):
                series[channel] = read_channel(
                    archive, indexed[key][channel]["member"], channel, number)
            cycle = series["Capacity"][0]
            if any(not np.array_equal(cycle, series[name][0]) for name in CHANNELS):
                raise ValueError(f"channel cycle alignment mismatch: {key}")
            capacity = series["Capacity"][1]
            ce, eoc, eod = (series[name][1] for name in ("CE", "EOC", "EOD"))
            if np.any(capacity <= 0):
                raise ValueError(f"nonpositive capacity: {key}")
            cell_id = f"{pack}_Cell{number:02d}"
            reference_first = float(capacity[0])
            reference_median5 = float(np.median(capacity[:5]))
            soh_first = capacity / reference_first
            soh_median5 = capacity / reference_median5
            first_event = first_confirmed_candidate(cycle, soh_first)
            median5_event = first_confirmed_candidate(cycle, soh_median5)
            large_recovery = np.flatnonzero(np.diff(capacity) > .05 * reference_first)
            event_index = (np.searchsorted(cycle, median5_event)
                           if median5_event is not None else None)
            recovered_above80 = (bool(np.any(soh_median5[event_index + 1:] > .82))
                                 if event_index is not None else False)
            meta = summary[key]
            cells.append({
                "dataset_id": "INL_NMC_44", "cell_id": cell_id,
                "pack_group": pack,
                "chemistry": indexed[key]["Capacity"]["chemistry"],
                "electrode_design": indexed[key]["Capacity"]["electrode_design"],
                "rows": len(cycle), "first_cycle": int(cycle[0]),
                "last_cycle": int(cycle[-1]),
                "source_experiment_end_cycle": meta["source_experiment_end_cycle"],
                "reference_first_ah": reference_first,
                "reference_median5_ah": reference_median5,
                "candidate80_first_reference_cycle": first_event,
                "candidate80_median5_reference_cycle": median5_event,
                "recovered_above82pct_after_candidate": recovered_above80,
                "capacity_jumps_over_5pct_reference": int(len(large_recovery)),
                "invalid_ce_count": int(np.count_nonzero((ce < 0) | (ce > 120))),
                "ce_review_count": int(np.count_nonzero((ce < 80) | (ce > 105))),
                "invalid_voltage_count": int(np.count_nonzero(
                    (eoc < 1.5) | (eoc > 5) | (eod < 1.5) | (eod > 5))),
                "metadata": meta,
                "label_status": "capacity_and_SOH_candidates_only; EOL80/RUL_not_training_approved",
            })
            for i, c in enumerate(cycle):
                diagnostics.append({
                    "dataset_id": "INL_NMC_44", "cell_id": cell_id,
                    "pack_group": pack, "cycle_index": int(c),
                    "capacity_ah": float(capacity[i]),
                    "coulombic_efficiency_percent": float(ce[i]),
                    "end_of_charge_voltage_v": float(eoc[i]),
                    "end_of_discharge_voltage_v": float(eod[i]),
                    "soh_first_reference_candidate": float(soh_first[i]),
                    "soh_median5_reference_candidate": float(soh_median5[i]),
                    "soh_label_approved": False,
                    "rul80_cycles": None,
                })
    return cells, diagnostics
