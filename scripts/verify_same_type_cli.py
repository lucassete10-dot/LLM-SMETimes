"""Check saved label-free CLI examples against frozen held-out predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prepare_same_type_forecast import save_json


def compare_prediction(prediction_file, arrays_file, cell_id, column):
    output = json.loads(prediction_file.read_text(encoding="utf-8"))
    cycles = np.asarray(output["future_cycles"], dtype=int)
    values = np.asarray(output["predicted_soh"], dtype=float)
    if len(cycles) != len(values) or not np.isfinite(values).all():
        raise ValueError(f"invalid CLI prediction in {prediction_file}")
    with np.load(arrays_file, allow_pickle=False) as arrays:
        heldout_cycle = arrays[f"{cell_id}__100__cycle"]
        expected = arrays[f"{cell_id}__100__{column}"]
    common, prediction_indices, expected_indices = np.intersect1d(
        cycles, heldout_cycle, return_indices=True)
    if len(common) == 0:
        raise ValueError("CLI result and held-out target cycles do not overlap")
    maximum = float(np.max(np.abs(values[prediction_indices] - expected[expected_indices])))
    if maximum > 1e-10:
        raise ValueError(f"saved CLI replay differs: {maximum}")
    return {"input": str(prediction_file.resolve()), "cell_id": cell_id,
            "compared_cycles": len(common), "max_absolute_difference": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--nasa-run", type=Path, required=True)
    args = parser.parse_args()
    checks = []
    for fold in ("MIT_repeat1", "XJTU_repeat1", "NASA_fold1"):
        folder = args.main_run / fold
        provenance = json.loads((folder / "example_provenance.json").read_text(encoding="utf-8"))
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        checks.append(compare_prediction(folder / "example_cli_prediction.json",
            folder / "test/predictions.npz", provenance["test_cell"], result["selected_method"]))
    checks.append(compare_prediction(args.nasa_run / "NASA_fold1/example_cli_prediction.json",
        args.nasa_run / "NASA_fold1/test/predictions.npz", "B0005", "prediction"))
    global_result = json.loads((args.nasa_run / "example_cli_prediction.json").read_text(encoding="utf-8"))
    if global_result["method"] != "voltage_ridge_exploratory" or not np.isfinite(global_result["predicted_soh"]).all():
        raise ValueError("invalid all-four NASA exploratory model output")
    save_json(args.main_run / "cli_verification.json", {"checks": checks,
        "all_four_nasa_exploratory_points": len(global_result["predicted_soh"]),
        "note": "all-four NASA weights include the example cell; only the refit-fold check is held-out"})
    print("VERIFIED", len(checks), "saved CLI-to-heldout prediction paths")


if __name__ == "__main__":
    main()
