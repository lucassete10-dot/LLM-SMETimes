"""Freeze classifications, label-free early signals and quality evidence."""

from collections import Counter
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import h5py
import numpy as np
from scipy.io import loadmat

from adapt_nasa_mat_v2 import prefix_features
from audit_mit_label_free_extraction_v2 import MIT_SOURCE
from battery_smetimes.forecast import type_key
from battery_v2_features import load_manifest
from prepare_mit import GRID, discharge_slice, feature_vector, read_ref
from prepare_xjtu import diagnostic, stage_features

ROOT = Path("data/interim/same_type_forecast")
MAX_ORIGIN = 100


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mit_prefix(mat, index, max_origin=MAX_ORIGIN):
    b = mat["batch"]
    numbers = np.asarray(mat[b["summary"][index, 0]]["cycle"]).reshape(-1).astype(int)
    data = mat[b["cycles"][index, 0]]
    rows, rejected = [], []
    if max_origin < 1:
        raise ValueError("source cycle budget must be positive")
    for j in np.flatnonzero(numbers <= max_origin):
        try:
            t, v, i, temp = [read_ref(mat, data[name][j, 0]) for name in ("t", "V", "I", "T")]
            if not (len(t) == len(v) == len(i) == len(temp)) or not np.isfinite(np.c_[t, v, i]).all():
                raise ValueError("inconsistent or nonfinite signal")
            if np.any(np.diff(t) * 60 < -1e-6):
                raise ValueError("nonmonotone time")
            run = discharge_slice(t, i, v)
            if run is None:
                raise ValueError("no 300-second main discharge")
            raw = feature_vector(t[run], i[run], v[run], temp[run], GRID)
            if np.isinf(raw).any():
                raise ValueError("infinite derived signal")
            rows.append((int(numbers[j]), raw))
        except ValueError as error:
            rejected.append({"cycle": int(numbers[j]), "reason": str(error)})
    return rows, rejected


def other_prefix(record):
    source = Path(record["measurement_source"])
    rows, rejected = [], []
    if record["dataset"] == "NASA":
        records = loadmat(source, variable_names=[record["cell_id"]], simplify_cells=True)[record["cell_id"]]["cycle"]
        count = 0
        for cycle in records:
            if cycle["type"] != "discharge":
                continue
            count += 1
            if count > MAX_ORIGIN:
                break
            try:
                raw = prefix_features(cycle["data"])
                if not np.isfinite(raw).all():
                    raise ValueError("nonfinite signal")
                rows.append((count, raw))
            except ValueError as error:
                rejected.append({"cycle": count, "reason": str(error)})
    else:
        records = loadmat(source, variable_names=["data"], simplify_cells=True)["data"]
        batch = int(record["group"].replace("batch", ""))
        for number, cycle in enumerate(records[:MAX_ORIGIN], 1):
            if not diagnostic(batch, cycle["description"]):
                rejected.append({"cycle": number, "reason": "not comparable full-discharge diagnostic; usage record, not corrupt data"})
                continue
            charge, discharge = stage_features(cycle, 0), stage_features(cycle, 2)
            if charge is None or discharge is None:
                rejected.append({"cycle": number, "reason": "incomplete/nonfinite 120-second signal prefix"})
                continue
            rows.append((number, np.r_[charge, discharge]))
    return rows, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-other-signals", type=Path, help="Verified NASA/XJTU signal cache; re-extract MIT")
    parser.add_argument("--mit-restoration-audit", type=Path)
    args = parser.parse_args()
    restoration_file = args.mit_restoration_audit
    if restoration_file is None and (ROOT / "latest_target_restoration.json").exists():
        restoration_file = Path(json.loads((ROOT / "latest_target_restoration.json").read_text())["file"])
    restored = {}
    if restoration_file is not None:
        for row in json.loads(restoration_file.read_text())["rows"]:
            if row["review_conclusion"] == "floating_point_time_reversal_below_1_microsecond":
                restored.setdefault(row["cell_id"], []).append(row)
    cached, cached_audit = {}, {}
    if args.reuse_other_signals:
        old_config = json.loads((args.reuse_other_signals / "config.json").read_text(encoding="utf-8"))
        for file in ("prepare_xjtu.py", "adapt_nasa_mat_v2.py"):
            if old_config["source_sha256"][file] != sha(file):
                raise ValueError("cached signal extraction code changed")
        cached = {r["cell_id"]: r for r in json.loads((args.reuse_other_signals / "manifest.json").read_text(encoding="utf-8"))["cells"]}
        cached_audit = {r["cell_id"]: r for r in json.loads((args.reuse_other_signals / "quality_audit.json").read_text(encoding="utf-8"))["by_cell"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = ROOT / stamp
    folder.mkdir(parents=True, exist_ok=False)
    print("PREPARE", folder, flush=True)
    records = [r for r in load_manifest()["cells"] if r["dataset"] in ("MIT", "XJTU", "NASA")]
    source_files = [__file__, "battery_smetimes/forecast.py", "prepare_mit.py", "prepare_xjtu.py",
                    "adapt_nasa_mat_v2.py", "battery_data_v2/manifest.json"]
    save_json(folder / "config.json", {"max_source_cycle": MAX_ORIGIN,
        "source_sha256": {str(p): sha(p) for p in source_files},
        "mit_restoration_audit": str(restoration_file.resolve()) if restoration_file else None,
        "mit_restoration_audit_sha256": sha(restoration_file) if restoration_file else None,
        "purpose": "signal-only prefix extraction; existing capacity contract used only for targets",
        "capacity_outlier_rule": "nonfinite/nonpositive targets excluded; existing local median flags retained for review"})
    written, audit = [], []
    handles = {}
    excluded = {"b2c7", "b2c8", "b2c9", "b2c15", "b2c16", "b3c37"}
    if excluded & {r["cell_id"] for r in records}:
        raise ValueError("known duplicates/collection-error cell entered manifest")
    try:
        for record in records:
            key, domain = record["cell_id"], record["dataset"]
            metadata = {name: record.get(name) for name in (
                "dataset", "group", "chemistry", "charge_protocol", "charge_rate_steps_c",
                "charge_rate_c", "discharge_rate_c", "diagnostic_discharge_rate_c")}
            if domain == "NASA":
                metadata["discharge_cutoff_v"] = {"B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5}[key]
            if domain == "MIT":
                batch = int(record["group"].replace("batch", ""))
                if batch not in handles:
                    handles[batch] = h5py.File(MIT_SOURCE[batch], "r")
                rows, rejected = mit_prefix(handles[batch], int(key.split("c")[1]))
            elif key in cached:
                old = cached[key]
                if sha(old["file"]) != old["sha256"] or sha(record["labels_file"]) != old["label_source_sha256"]:
                    raise ValueError("cached signal/label fingerprints differ")
                with np.load(old["file"]) as value:
                    rows = list(zip(value["cycle"].tolist(), value["raw_features"]))
                rejected = cached_audit[key]["prefix_exclusions"]
            else:
                rows, rejected = other_prefix(record)
            cycle = np.asarray([r[0] for r in rows], dtype=np.int64)
            raw = np.asarray([r[1] for r in rows], dtype=np.float32)
            if not len(cycle) or np.any(np.diff(cycle) <= 0):
                raise ValueError(f"{key}: missing or nonmonotone early signals")
            with np.load(record["labels_file"]) as labels:
                lc, q = labels["cycle"].astype(np.int64), labels["capacity_ah"].astype(float)
                soh = labels["soh"].astype(float)
                available = labels["soh_available"].astype(bool)
                review = labels["capacity_review_flag"].astype(bool)
            reference = float(record["reference_capacity_ah"])
            additions = [r for r in restored.get(key, []) if r["cycle"] not in set(lc.tolist())]
            if additions:
                lc = np.r_[lc, [r["cycle"] for r in additions]].astype(np.int64)
                added_q = np.array([r["capacity_ah"] for r in additions], dtype=float)
                q = np.r_[q, added_q]
                soh = np.r_[soh, added_q / reference]
                available = np.r_[available, np.ones(len(additions), dtype=bool)]
                review = np.r_[review, np.zeros(len(additions), dtype=bool)]
                order = np.argsort(lc)
                lc, q, soh, available, review = [a[order] for a in (lc, q, soh, available, review)]
            bad = ~np.isfinite(q) | (q <= 0)
            if reference <= 0 or not np.isfinite(reference) or np.any(np.diff(lc) <= 0):
                raise ValueError(f"{key}: invalid capacity reference/cycle axis")
            if not np.allclose(soh[available], q[available] / reference, rtol=1e-5, atol=1e-5):
                raise ValueError(f"{key}: SOH/reference mismatch")
            soh[~available | bad] = np.nan
            file = folder / f"{key}.npz"
            np.savez_compressed(file, cycle=cycle, raw_features=raw, label_cycle=lc,
                soh=soh, capacity_review_flag=review)
            detail = {"cell_id": key, "metadata": metadata, "type": type_key(metadata),
                "file": str(file.resolve()), "sha256": sha(file), "label_source": record["labels_file"],
                "label_source_sha256": sha(record["labels_file"]), "raw_source": record["measurement_source"],
                "raw_source_sha256_from_existing_manifest": record.get("measurement_sha256"),
                "reference_capacity_ah": reference, "reference_rule": record["reference_rule"],
                "restored_source_label_cycles": [r["cycle"] for r in additions],
                "signal_observations": len(cycle), "label_observations": int(np.isfinite(soh).sum())}
            written.append(detail)
            audit.append({"cell_id": key, "prefix_exclusions": rejected,
                "restored_source_label_cycles": [r["cycle"] for r in additions],
                "bad_target_cycles": lc[bad].tolist(), "reference_only_cycles": lc[~available].tolist(),
                "capacity_review_cycles_kept": lc[review].tolist(),
                "missing_signal_features": int(np.isnan(raw).sum())})
            save_json(folder / "progress.json", {"completed_cells": len(written), "expected_cells": len(records)})
            print("SAVED", key, "signals", len(cycle), "targets", detail["label_observations"], flush=True)
        legacy = {}
        for batch in (1, 2, 3):
            path = Path(r"D:\LLM\MIT\processed") / (f"batch{batch}/manifest.json" if batch > 1 else "manifest.json")
            old = json.loads(path.read_text(encoding="utf-8"))
            legacy[str(batch)] = {"manifest": str(path), "sha256": sha(path), "reasons": old["excluded_reasons"]}
        save_json(folder / "quality_audit.json", {"known_cell_exclusions": {
            "MIT_Batch2_continuations_of_Batch1": sorted(excluded - {"b3c37"}),
            "MIT_author_documented_collection_error": ["b3c37"]},
            "legacy_MIT_label_exclusions": legacy,
            "legacy_XJTU_target_rules": "comparable full diagnostics; capacity 0.8<Q<2.4 Ah; reference median first five 1C/0.5C diagnostics; source life unused",
            "outlier_review_policy": "review flag alone never removes a target or a cell",
            "by_cell": audit})
        save_json(folder / "manifest.json", {"schema": 1, "source_cycle_limit": MAX_ORIGIN,
            "counts": dict(Counter(r["metadata"]["dataset"] for r in written)),
            "type_counts": dict(Counter(r["type"] for r in written)), "cells": written})
        save_json(ROOT / "latest.json", {"folder": str(folder.resolve())})
        print("COMPLETE", folder, flush=True)
    except BaseException:
        save_json(folder / "failure.json", {"traceback": traceback.format_exc(), "completed_cells": len(written)})
        raise
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
