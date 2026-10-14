"""Predict from saved models and label-free causal MIT feature histories.

Input NPZ fields: cycle [N], signal_history [N,153], reference_count scalar.
Features must use the same 300-second MIT signal/causal-reference contract
as training. No capacity, true SOH, endpoint or remaining-life input is read.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from battery_smetimes.models.trajectory import anchored_soh, log_rul_interval
from scripts.diagnose_mit_estimated_health_rul import predict_soh, predicted_health_features


def predict_history(cycle, signal_history, reference_count, fold_directory):
    cycle = np.asarray(cycle, dtype=float)
    history = np.asarray(signal_history, dtype=np.float32)
    if (history.ndim != 2 or len(history) != len(cycle) or
            not 1 <= reference_count < len(cycle) or np.any(np.diff(cycle) <= 0) or
            not np.isfinite(cycle).all() or np.isinf(history).any()):
        raise ValueError("valid established-reference causal history required")
    folder = Path(fold_directory)
    soh_model = joblib.load(folder / "soh_predictor.joblib")
    rul_model = joblib.load(folder / "rul_predictor.joblib")
    shape = json.loads((folder / "calibrated_shape.json").read_text(encoding="utf-8"))["selected_shape"]
    radius = json.loads((folder / "interval_calibration.json").read_text(encoding="utf-8"))["radius"]
    signals = np.column_stack((cycle, history)).astype(np.float32)
    soh = predict_soh(soh_model, signals)
    proxy = SimpleNamespace(cycle=cycle, reference_count=int(reference_count))
    health = predicted_health_features(proxy, soh)
    x = np.column_stack((signals, health[:, 1:]))[-1:]
    remaining = float(np.expm1(rul_model["model"].predict(rul_model["imputer"].transform(x)))[0])
    horizons = np.unique(np.r_[np.linspace(0, remaining, 201), [10, 50, 100]])
    interval = log_rul_interval(remaining, radius)
    return {"origin_cycle": float(cycle[-1]), "estimated_current_soh": float(soh[-1]),
            "rul_cycles": remaining, "eol_cycle": float(cycle[-1] + remaining),
            "rul_interval_cycles": list(interval), "nominal_joint_coverage": .9,
            "shape": shape, "threshold": .8, "pre_eol_anchor_projection": bool(soh[-1] <= .8001),
            "forecast_cycle": (cycle[-1] + horizons).tolist(),
            "forecast_soh": anchored_soh(soh[-1], remaining, horizons, shape).tolist(),
            "scope": "pre-EOL MIT Batch-2 development model; calibrated coverage is empirical, not guaranteed on new domains"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output filename to preserve previous predictions")
    with np.load(args.features, allow_pickle=False) as source:
        result = predict_history(source["cycle"], source["signal_history"],
                                 int(source["reference_count"].item()), args.fold_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
