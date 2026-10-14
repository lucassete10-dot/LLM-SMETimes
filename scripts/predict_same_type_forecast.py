"""Forecast a new signal prefix with a saved same-type model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from battery_smetimes.forecast import predict_curve, prefix_vector, type_key
from scripts.prepare_same_type_forecast import save_json
from scripts.train_nasa_voltage_forecast import input_matrix, voltage_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True, help="NPZ: cycle and raw_features only")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON: known dataset/group/protocol parameters")
    parser.add_argument("--origin", type=int, default=100)
    parser.add_argument("--end-cycle", type=int, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("output must be a new file in an existing directory")
    if args.step <= 0 or args.end_cycle <= args.origin:
        raise ValueError("positive step and future end-cycle required")
    bundle = joblib.load(args.model)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata["dataset"] != bundle["dataset"]:
        raise ValueError("model and input dataset/type differ")
    with np.load(args.signals, allow_pickle=False) as data:
        if set(data.files) != {"cycle", "raw_features"}:
            raise ValueError("input must contain only cycle/raw_features; no target fields")
        cycle, raw = data["cycle"], data["raw_features"]
    future = np.arange(args.origin + args.step, args.end_cycle + 1, args.step)
    if bundle.get("model_schema") == "nasa_voltage_ridge_v1":
        if metadata["dataset"] != "NASA":
            raise ValueError("NASA voltage model requires NASA input")
        # Apply the same prefix-shape, order, and minimum-history checks as the
        # general forecaster; no future signal or target is consulted.
        prefix_vector(cycle, raw, metadata, args.origin, [type_key(metadata)])
        voltage = voltage_prefix({"cycle": cycle, "raw": raw}, args.origin)
        prediction = np.clip(bundle["model"].predict(bundle["scaler"].transform(
            input_matrix(voltage, args.origin, future))), 0, 1.2)
        support = np.sum(np.asarray(bundle["training_support_end_cycles"])[:, None]
                         >= future[None, :], axis=0)
        method = "voltage_ridge_exploratory"
    else:
        prediction, support = predict_curve(bundle, cycle, raw, metadata, args.origin,
            future, bundle["selected_method"])
        method = bundle["selected_name"]
    save_json(args.output, {"type": type_key(metadata), "origin": args.origin,
        "evaluated_origins_source_cycles": list(bundle["origins"]),
        "origin_in_evaluated_set": args.origin in bundle["origins"],
        "soh_unit": "capacity/reference ratio (1.0 = 100%)",
        "future_cycles": future.tolist(), "predicted_soh": prediction.tolist(),
        "same_type_training_curves_covering_cycle": support.tolist(),
        "model": str(args.model.resolve()), "method": method,
        "future_assumption": "same known operating strategy continues",
        "beyond_training_support": bool(np.any(support == 0)),
        "nasa_voltage_model_is_exploratory": bundle.get("model_schema") == "nasa_voltage_ridge_v1"})
    print("SAVED", args.output, len(future), "future SOH points")


if __name__ == "__main__":
    main()
