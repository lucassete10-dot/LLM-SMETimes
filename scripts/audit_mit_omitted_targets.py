"""Check omitted positive MIT capacity labels against raw discharge integration."""
import json
from pathlib import Path
from datetime import datetime, timezone
import h5py
import numpy as np
from audit_mit_label_free_extraction_v2 import MIT_SOURCE
from prepare_mit import read_ref, discharge_slice
from scripts.prepare_same_type_forecast import ROOT, save_json


def main():
    findings = json.loads((ROOT / "inherited_target_audit_20260929T120320_622528Z.json").read_text())["omitted_source_records"]
    rows = []
    for batch, path in MIT_SOURCE.items():
        with h5py.File(path, "r") as mat:
            for finding in findings:
                if not finding["needs_review"] or not finding["cell_id"].startswith(f"b{batch}c"):
                    continue
                index = int(finding["cell_id"].split("c")[1])
                summary = mat[mat["batch"]["summary"][index, 0]]
                number = np.asarray(summary["cycle"]).reshape(-1).astype(int)
                j = int(np.flatnonzero(number == finding["cycle"])[0])
                cycles = mat[mat["batch"]["cycles"][index, 0]]
                t, i, v, qd = [read_ref(mat, cycles[field][j, 0]) for field in ("t", "I", "V", "Qd")]
                discharge = i < -2
                dt, di, dv = t[discharge], i[discharge], v[discharge]
                integrated = float(np.trapezoid(-di, dt * 60) / 3600) if len(dt) > 1 else None
                run = discharge_slice(t, i, v)
                gap = float(np.min(np.diff((t[run] - t[run[0]]) * 60))) if run is not None else None
                reason = ("capacity_counter_nonzero_at_main_discharge_start" if finding["capacity_ah"] > 1.25 and discharge.any() and qd[discharge][0] > .1 else
                          "summary_capacity_exceeds_main_discharge_by_more_than_50_percent" if integrated is not None and finding["capacity_ah"] > 1.5 * integrated and finding["capacity_ah"] > 1.25 else
                          "no_300_second_discharge_capture" if run is None else
                          "floating_point_time_reversal_below_1_microsecond" if gap is not None and -1e-6 <= gap < 0 else "unresolved")
                rows.append({**finding, "discharge_samples": int(discharge.sum()),
                    "review_conclusion": reason, "main_segment_min_time_step_s": gap,
                    "duration_s": float((dt[-1] - dt[0]) * 60) if len(dt) else None,
                    "end_voltage_v": float(dv[-1]) if len(dv) else None,
                    "integrated_discharge_ah": integrated,
                    "summary_integration_relative_difference": abs(integrated - finding["capacity_ah"]) / finding["capacity_ah"] if integrated else None,
                    "raw_qd_max_ah": float(np.max(qd)),
                    "raw_discharge_qd_start_end": [float(qd[discharge][0]), float(qd[discharge][-1])] if discharge.any() else None})
    output = ROOT / ("omitted_positive_mit_trace_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
    save_json(output, {"rows": rows})
    save_json(ROOT / "latest_target_restoration.json", {"file": str(output.resolve())})
    print("SAVED", output)
    for row in rows:
        print(row["cell_id"], row["cycle"], row["capacity_ah"], row["integrated_discharge_ah"], row["end_voltage_v"], row["discharge_samples"])


if __name__ == "__main__":
    main()
