"""Replay raw MAT adapters against prepared, quality-filtered forecast inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.extract_forecast_prefix import extract
from scripts.prepare_same_type_forecast import save_json
from scripts.train_same_type_forecast import load_data, OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    manifest, cells = load_data(Path(config["data_folder"]), config["integrity_file"])
    records = {r["cell_id"]: r for r in manifest["cells"]}
    examples = ("b1c10", "b2c12", "b3c0", "b1_2C_battery-1", "b6_Sim_satellite_battery-1", "B0018")
    checked = []
    for key in examples:
        c, r = cells[key], records[key]
        options = {"cell_index": int(key.split("c")[1])} if c["metadata"]["dataset"] == "MIT" else (
            {"variable": key} if c["metadata"]["dataset"] == "NASA" else {})
        cycle, raw = extract(Path(r["raw_source"]), c["metadata"], 100, **options)
        np.testing.assert_array_equal(cycle, c["cycle"])
        np.testing.assert_allclose(raw, c["raw"], rtol=0, atol=1e-7, equal_nan=True)
        checked.append({"cell_id": key, "observations": len(cycle), "max_cycle": int(cycle[-1]),
                        "excluded_interrupted_input_cycles": c["excluded_input_cycles"]})
        print("VERIFIED", key, len(cycle), flush=True)
    save_json(args.run / "raw_input_verification.json", {"checked": checked,
        "scope": "representatives including interrupted MIT input, malformed source policy, sparse XJTU diagnostics, and NASA",
        "target_fields_accessed_by_adapter": False})


if __name__ == "__main__":
    main()
