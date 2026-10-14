"""Freeze an XJTU 80%-SOH RUL challenger for a later independent test."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from battery_v2_features import load_all, make_inference_features
from evaluate_paper_inspired_rul_v2 import OUT as CV, features, fit, predict
from train_rul_v2 import MIN_HISTORY


ROOT = CV.parent
MODEL = ROOT / "xjtu_rul_time_gated_log_candidate.joblib"
MANIFEST = ROOT / "xjtu_rul_candidate_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_xjtu_rul(bundle, raw, cycle):
    cell = make_inference_features(raw, cycle, reference_count=5)
    if cell.raw.shape[1] != bundle["raw_feature_count"]:
        raise ValueError("XJTU feature schema differs from the trained protocol")
    result = np.asarray(predict(bundle["model"], features(cell), "gated_log"), dtype=np.float32)
    result[:max(MIN_HISTORY, cell.reference_count)] = np.nan
    return result


def train():
    if MODEL.exists() or MANIFEST.exists():
        raise FileExistsError("RUL challenger already exists; verify it instead of overwriting")
    report = json.loads(CV.read_text(encoding="utf-8"))
    if set(report["domains"]) != {"NASA", "CALCE", "MIT", "XJTU"}:
        raise ValueError("complete four-domain RUL development comparison is required")
    source, cells = load_all()
    ids = [row["cell_id"] for row in source["cells"] if row["dataset"] == "XJTU"
           and cells[row["cell_id"]].endpoint_cycle is not None]
    matrices = {key: features(cells[key]) for key in ids}
    model = fit(cells, ids, matrices, "gated_log")
    bundle = {"kind": "XJTU time-gated direct RUL with log1p target",
              "endpoint_soh": .8,
              "raw_feature_count": cells[ids[0]].raw.shape[1],
              "training_cell_ids": ids, "model": model}
    joblib.dump(bundle, MODEL, compress=3)
    record = {"status": "development-selected RUL challenger, awaiting independent test",
              "primary_selection": "gated_log: improves both XJTU equal-fold MAE and cycle-50 EOL MAPE",
              "candidate_sha256": sha256(MODEL),
              "development_cv_sha256": sha256(CV),
              "source_manifest_sha256": sha256("battery_data_v2/manifest.json"),
              "observed_endpoint_training_cells": len(ids),
              "censoring": "censored cells excluded; no guessed endpoint labels",
              "warning": "development selection is not independent validation"}
    MANIFEST.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("FROZEN", MODEL, len(ids), "observed-endpoint cells", flush=True)


def verify():
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key, path in (("candidate_sha256", MODEL),
                      ("development_cv_sha256", CV),
                      ("source_manifest_sha256", Path("battery_data_v2/manifest.json"))):
        if record[key] != sha256(path):
            raise ValueError(f"frozen RUL challenger dependency changed: {path}")
    print("VERIFIED", MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else train()
