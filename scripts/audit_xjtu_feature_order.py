"""Verify all 46 named XJTU columns against independently saved raw prefixes."""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from prepare_xjtu import FEATURE_NAMES


WAVE = Path("data/interim/xjtu_waveforms")
OUT = Path("battery_models_v3/xjtu_charge_stage_a/feature_order_audit.json")


def reconstructed_stage(wave):
    sampled = wave[:, :3, ::30].reshape(len(wave), 15).astype(np.float64)
    v, current, temperature = np.split(sampled, 3, axis=1)
    seconds = np.arange(0, 121, 30, dtype=np.float64)
    extra = np.column_stack((
        v[:, -1] - v[:, 0], np.mean(v, axis=1),
        (v[:, -1] - v[:, 0]) / 120,
        np.mean(current, axis=1), np.std(current, axis=1),
        temperature[:, -1] - temperature[:, 0],
        np.mean(temperature, axis=1),
        np.trapezoid(np.abs(current), seconds, axis=1) / 3600,
    ))
    return np.column_stack((sampled, extra)).astype(np.float32)


def main():
    if len(FEATURE_NAMES) != 46 or len(set(FEATURE_NAMES)) != 46:
        raise ValueError("XJTU feature names are incomplete/duplicated")
    if any(not name.startswith("charge_") for name in FEATURE_NAMES[:23]) or \
       any(not name.startswith("discharge_") for name in FEATURE_NAMES[23:]):
        raise ValueError("XJTU name stages do not match 23+23 raw storage")
    records = [row for row in load_manifest()["cells"] if row["dataset"] == "XJTU"]
    errors, cycles = [], 0
    for record in records:
        cell = load_cell(record)
        with np.load(WAVE / f"{cell.cell_id}.npz") as source:
            if not np.array_equal(source["cycle"], cell.cycle):
                raise ValueError(f"cycle mismatch: {cell.cell_id}")
            wave = source["waveform"]
        reconstructed = np.concatenate((reconstructed_stage(wave[:, 0]),
                                        reconstructed_stage(wave[:, 1])), axis=1)
        absolute = np.abs(reconstructed - cell.raw)
        if not np.allclose(reconstructed, cell.raw, rtol=1e-5, atol=1e-5,
                           equal_nan=True):
            raise ValueError(f"raw 46-column contract mismatch: {cell.cell_id}")
        errors.append(float(np.nanmax(absolute)))
        cycles += len(cell.cycle)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "all existing XJTU raw feature columns independently checked",
        "cells": len(records), "diagnostic_cycles": cycles,
        "feature_names": FEATURE_NAMES,
        "charge_columns_0_based": [0, 22],
        "discharge_columns_0_based": [23, 45],
        "max_abs_column_difference": max(errors),
        "note": "old feature-name lists grouped both stages' sample points before extras; numerical model inputs were not reordered",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Verified", len(records), "cells and", cycles,
          "diagnostics; max absolute difference", max(errors))


if __name__ == "__main__":
    main()
