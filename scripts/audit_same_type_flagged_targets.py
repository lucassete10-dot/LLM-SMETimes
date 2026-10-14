"""Inspect original MIT traces at pre-existing capacity-review flags."""

from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from audit_mit_label_free_extraction_v2 import MIT_SOURCE
from battery_v2_features import load_manifest
from prepare_mit import read_ref
from scripts.prepare_same_type_forecast import save_json


def main():
    records = [r for r in load_manifest()["cells"] if r["dataset"] == "MIT"]
    findings = []
    for batch, source in MIT_SOURCE.items():
        with h5py.File(source, "r") as mat:
            for record in records:
                if record["group"] != f"batch{batch}":
                    continue
                with np.load(record["labels_file"]) as labels:
                    flagged = labels["cycle"][labels["capacity_review_flag"]]
                if not len(flagged):
                    continue
                index = int(record["cell_id"].split("c")[1])
                summary = mat[mat["batch"]["summary"][index, 0]]
                cycles = mat[mat["batch"]["cycles"][index, 0]]
                number = np.asarray(summary["cycle"]).reshape(-1).astype(int)
                q = np.asarray(summary["QDischarge"]).reshape(-1)
                for event in flagged:
                    j = int(np.flatnonzero(number == event)[0])
                    observations = []
                    for position in range(max(0, j - 1), min(len(number), j + 2)):
                        t, i, v, temp, qd = [read_ref(mat, cycles[field][position, 0]) for field in ("t", "I", "V", "T", "Qd")]
                        discharge = np.flatnonzero(i < -2)
                        observations.append({"cycle": int(number[position]), "summary_capacity_ah": float(q[position]),
                            "max_curve_discharge_ah": float(np.nanmax(qd)),
                            "cc_discharge_duration_s": float(60 * (t[discharge[-1]] - t[discharge[0]])),
                            "cc_final_voltage_v": float(v[discharge[-1]]),
                            "cc_min_voltage_v": float(np.min(v[discharge])),
                            "cc_mean_abs_current_a": float(np.mean(np.abs(i[discharge]))),
                            "cc_mean_temperature_c": float(np.nanmean(temp[discharge]))})
                    findings.append({"cell_id": record["cell_id"], "flagged_cycle": int(event), "neighbor_traces": observations,
                        "action": "review only; no automatic exclusion from a local capacity flag"})
                    print(record["cell_id"], int(event), json.dumps(observations), flush=True)
    root = Path("data/interim/same_type_forecast")
    name = datetime.now(timezone.utc).strftime("flagged_target_trace_audit_%Y%m%dT%H%M%SZ.json")
    save_json(root / name, {"findings": findings, "policy": "trace inspection; no model-error-based filtering"})
    print("SAVED", root / name)


if __name__ == "__main__":
    main()
