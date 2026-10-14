"""Train and freeze the MIT/XJTU paper-inspired SOH challenger.

NASA/CALCE keep their prior selected models. This challenger remains separate
from the existing frozen production/development candidate until independent
test data are supplied.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from battery_v2_features import load_all, make_inference_features
from evaluate_paper_inspired_soh_v2 import OUT, fit_gated
from evaluate_relative_soh_features_v2 import fit_view, predict_view
from paper_inspired_battery_v2 import time_gated_matrix


MODEL = OUT / "mit_xjtu_candidate.joblib"
MANIFEST = OUT / "candidate_manifest.json"
DOMAIN_METHOD = {"NASA": "unchanged_baseline", "CALCE": "unchanged_baseline",
                 "MIT": "time_gated", "XJTU": "raw_time_gated_half"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_challenger(bundle, dataset, raw, cycle):
    dataset = dataset.upper()
    if dataset not in ("MIT", "XJTU"):
        raise ValueError("challenger is only trained for MIT and XJTU protocols")
    features = make_inference_features(raw, cycle, 1 if dataset == "MIT" else 5)
    if features.raw.shape[1] != bundle[dataset]["raw_feature_count"]:
        raise ValueError("signal feature schema differs from the trained protocol")
    matrix = time_gated_matrix(features)
    imputer, model = bundle[dataset]["gated"]
    gated = model.predict(imputer.transform(matrix))
    if dataset == "XJTU":
        gated = .5 * (gated + predict_view(bundle[dataset]["raw"], features, "raw_only"))
    result = np.asarray(gated, dtype=np.float32)
    result[:features.reference_count] = np.nan
    return result


def train():
    if MODEL.exists() or MANIFEST.exists():
        raise FileExistsError("challenger already exists; verify it instead of overwriting")
    cv = json.loads((OUT / "development_cv.json").read_text(encoding="utf-8"))
    if not all(domain in cv["domains"] for domain in DOMAIN_METHOD):
        raise ValueError("complete four-domain development comparison is required")
    manifest, cells = load_all()
    bundle = {}
    for domain in ("MIT", "XJTU"):
        ids = [row["cell_id"] for row in manifest["cells"] if row["dataset"] == domain]
        matrices = {key: time_gated_matrix(cells[key]) for key in ids}
        value = {"raw_feature_count": cells[ids[0]].raw.shape[1],
                 "training_cell_ids": ids, "gated": fit_gated(cells, ids, matrices)}
        if domain == "XJTU":
            value["raw"] = fit_view(cells, ids, "raw_only")
        bundle[domain] = value
        print("TRAINED", domain, len(ids), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL, compress=3)
    record = {"status": "development-selected candidate, awaiting independent test",
              "methods": DOMAIN_METHOD,
              "candidate_sha256": sha256(MODEL),
              "development_cv_sha256": sha256(OUT / "development_cv.json"),
              "source_manifest_sha256": sha256("battery_data_v2/manifest.json"),
              "inputs": "protocol-matched, capacity-free causal signal features",
              "warning": "do not treat development CV as an unbiased final test"}
    MANIFEST.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("FROZEN", MODEL, flush=True)


def verify():
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key, path in (("candidate_sha256", MODEL),
                      ("development_cv_sha256", OUT / "development_cv.json"),
                      ("source_manifest_sha256", Path("battery_data_v2/manifest.json"))):
        if record[key] != sha256(path):
            raise ValueError(f"frozen challenger dependency changed: {path}")
    print("VERIFIED", MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else train()
