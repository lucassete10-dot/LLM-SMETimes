"""Label-independent discharge integrity scan across the accepted MIT cells."""

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from audit_mit_label_free_extraction_v2 import MIT_SOURCE
from battery_v2_features import load_manifest
from prepare_mit import read_ref
from scripts.prepare_same_type_forecast import ROOT, save_json


def discontinuity(t_minutes, current):
    t, i = np.asarray(t_minutes), np.asarray(current)
    if len(t) != len(i) or not np.isfinite(t).all() or not np.isfinite(i).all():
        return "nonfinite_or_misaligned_time_current", None
    gap = np.diff(t) * 60
    # Source step boundaries contain ~1e-11 s floating-point reversals.
    if np.any(gap < -1e-6):
        return "nonmonotone_time", None
    continuous_discharge = (i[:-1] < -2) & (i[1:] < -2)
    largest = float(np.max(gap[continuous_discharge])) if np.any(continuous_discharge) else 0.
    if largest > 60:
        return "gap_over_60s_between_adjacent_4C_discharge_samples", largest
    discharge = np.flatnonzero(i < -2)
    if len(discharge) and 60 * (t[discharge[-1]] - t[discharge[0]]) > 3600:
        return "4C_discharge_episode_spans_over_1_hour_interrupted_protocol", largest
    return None, largest


def main():
    records = [r for r in load_manifest()["cells"] if r["dataset"] == "MIT"]
    issues, scanned = [], 0
    for batch, path in MIT_SOURCE.items():
        with h5py.File(path, "r") as mat:
            for record in records:
                if record["group"] != f"batch{batch}":
                    continue
                index = int(record["cell_id"].split("c")[1])
                numbers = np.asarray(mat[mat["batch"]["summary"][index, 0]]["cycle"]).reshape(-1).astype(int)
                cycles = mat[mat["batch"]["cycles"][index, 0]]
                for j, number in enumerate(numbers):
                    t, i = read_ref(mat, cycles["t"][j, 0]), read_ref(mat, cycles["I"][j, 0])
                    reason, gap = discontinuity(t, i)
                    scanned += 1
                    if reason:
                        issues.append({"cell_id": record["cell_id"], "cycle": int(number),
                            "reason": reason, "largest_adjacent_discharge_sample_gap_s": gap,
                            "action": "exclude interrupted/invalid diagnostic capacity as a regular-cycle training target; keep cell"})
                print("SCANNED", record["cell_id"], "issues_total", len(issues), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    file = ROOT / f"mit_discharge_integrity_{stamp}.json"
    save_json(file, {"scanned_cells": len(records), "scanned_source_cycles": scanned,
        "rule": "finite/aligned time/current; reversals exceeding 1 microsecond; adjacent I<-2 A sample gap >60 s; or first-to-last I<-2 A span >3600 s (interrupted 4C episode)",
        "capacity_or_prediction_used_in_rule": False, "issues": issues})
    save_json(ROOT / "latest_integrity.json", {"file": str(file.resolve())})
    print("SAVED", file, "issues", len(issues))


if __name__ == "__main__":
    main()
