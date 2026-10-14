"""Account for inherited target screens against original source capacities."""

from collections import Counter
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from audit_mit_label_free_extraction_v2 import MIT_SOURCE
from prepare_mit import read_ref
from prepare_xjtu import diagnostic, stage_features
from scripts.audit_mit_discharge_integrity import discontinuity
from scripts.prepare_same_type_forecast import ROOT, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("MIT", "XJTU", "all"), default="all")
    args = parser.parse_args()
    folder = Path(json.loads((ROOT / "latest.json").read_text())["folder"])
    records = json.loads((folder / "manifest.json").read_text())["cells"]
    if args.dataset != "all":
        records = [r for r in records if r["metadata"]["dataset"] == args.dataset]
    findings, reference_checks = [], []
    for batch, source in MIT_SOURCE.items():
        with h5py.File(source, "r") as mat:
            for record in records:
                if record["metadata"]["dataset"] != "MIT" or record["metadata"]["group"] != f"batch{batch}":
                    continue
                index = int(record["cell_id"].split("c")[1])
                summary = mat[mat["batch"]["summary"][index, 0]]
                numbers = np.asarray(summary["cycle"]).reshape(-1).astype(int)
                capacities = np.asarray(summary["QDischarge"]).reshape(-1)
                cycles = mat[mat["batch"]["cycles"][index, 0]]
                with np.load(record["file"]) as saved:
                    known = set(saved["label_cycle"].tolist())
                for j, (number, capacity) in enumerate(zip(numbers, capacities)):
                    if int(number) in known:
                        continue
                    t, current = [read_ref(mat, cycles[field][j, 0]) for field in ("t", "I")]
                    reason = "nonpositive_or_nonfinite_capacity" if not np.isfinite(capacity) or capacity <= 0 else discontinuity(t, current)[0]
                    findings.append({"dataset": "MIT", "cell_id": record["cell_id"], "cycle": int(number),
                        "capacity_ah": float(capacity) if np.isfinite(capacity) else None,
                        "signal_integrity_reason": reason,
                        "needs_review": reason is None})
    for record in records:
        if record["metadata"]["dataset"] != "XJTU":
            continue
        summary = loadmat(record["raw_source"], variable_names=["summary"], simplify_cells=True)["summary"]
        capacity = np.asarray(summary["discharge_capacity_Ah"]).reshape(-1)
        batch = int(record["metadata"]["group"].replace("batch", ""))
        with np.load(record["file"]) as saved:
            known = set(saved["label_cycle"].tolist())
        eligible = [float(capacity[j - 1]) for j in sorted(known)]
        # A range screen can only remove values outside its range. Avoid reading
        # all dense waveforms if every omitted capacity lies inside the range.
        suspicious = [j for j, q in enumerate(capacity, 1) if j not in known
                      and (not np.isfinite(q) or not .8 < q < 2.4)]
        data = loadmat(record["raw_source"], variable_names=["data"], simplify_cells=True)["data"] if suspicious else None
        for j in suspicious:
            cycle, q = data[j - 1], capacity[j - 1]
            if not diagnostic(batch, cycle["description"]):
                continue
            reason = "nonpositive_or_nonfinite_capacity" if not np.isfinite(q) or q <= 0 else None
            if reason is None and (stage_features(cycle, 0) is None or stage_features(cycle, 2) is None):
                reason = "incomplete_signal_prefix_in_legacy_export"
            findings.append({"dataset": "XJTU", "cell_id": record["cell_id"], "cycle": j,
                "capacity_ah": float(q) if np.isfinite(q) else None,
                "signal_integrity_reason": reason, "needs_review": reason is None})
        reference = float(np.median(eligible[:5]))
        if not np.isclose(reference, record["reference_capacity_ah"], rtol=1e-6):
            raise ValueError("source reference mismatch: " + record["cell_id"])
        reference_checks.append({"cell_id": record["cell_id"], "reference_capacity_ah": reference})
        del data
        print("AUDITED", record["cell_id"], flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / f"inherited_target_audit_{stamp}.json"
    save_json(output, {"source_prepared_data": str(folder), "omitted_source_records": findings,
        "scope": "MIT omitted source labels; XJTU inherited capacity-range screens and reference consistency. XJTU in-range omitted prefixes remain documented in the extraction audit.",
        "reason_counts": dict(Counter((r["signal_integrity_reason"] or "needs_review") for r in findings)),
        "xjtu_references_recomputed_from_raw": reference_checks})
    print("SAVED", output, "needs_review", sum(r["needs_review"] for r in findings), flush=True)


if __name__ == "__main__":
    main()
