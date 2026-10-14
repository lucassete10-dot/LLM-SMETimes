"""Build per-cell label-free XJTU 120-second waveform NPZs from original MAT."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parsers.xjtu import extract_mat_waveforms  # noqa: E402


OUT = ROOT / "data/interim/xjtu_waveforms"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / "battery_data_v2/manifest.json").read_text(encoding="utf-8"))
    records = [row for row in manifest["cells"] if row["dataset"] == "XJTU"]
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {"dataset_id": "XJTU", "source_variable_loaded": "data only",
             "label_source_loaded": False,
             "channels": ["voltage_V", "current_A", "temperature_C", "throughput_Ah"],
             "grid_seconds": "0 through 120 inclusive at 1-second intervals",
             "cells": []}
    for record in records:
        cell_id = record["cell_id"]
        source = Path(record["measurement_source"])
        batch = int(record["group"].removeprefix("batch"))
        target = OUT / (cell_id + ".npz")
        if target.exists():
            with np.load(target) as existing:
                cycle = existing["cycle"]
                waveform = existing["waveform"]
            if waveform.shape != (len(cycle), 2, 4, 121):
                raise ValueError(f"existing waveform shape mismatch: {target}")
        else:
            cycle, waveform = extract_mat_waveforms(source, batch)
            temporary = target.with_suffix(".npz.part")
            with temporary.open("wb") as output:
                np.savez_compressed(output, cycle=cycle, waveform=waveform)
            os.replace(temporary, target)
        with np.load(record["labels_file"]) as labels:
            label_cycle = labels["cycle"].astype(np.int64)
        if not np.array_equal(cycle, label_cycle):
            raise ValueError(f"{cell_id}: label-free waveform/label cycle mismatch")
        if not np.isfinite(waveform).all():
            raise ValueError(f"{cell_id}: nonfinite waveform")
        audit["cells"].append({"cell_id": cell_id, "group": record["group"],
                               "source_path": str(source), "source_sha256": sha256(source),
                               "waveform_path": str(target),
                               "diagnostics": len(cycle),
                               "first_cycle": int(cycle[0]),
                               "last_cycle": int(cycle[-1])})
        print("WAVEFORM", cell_id, len(cycle), flush=True)
    if len(audit["cells"]) != 55:
        raise ValueError("expected 55 XJTU source cells")
    (OUT / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("VERIFIED", len(audit["cells"]), "cells", flush=True)


if __name__ == "__main__":
    main()
