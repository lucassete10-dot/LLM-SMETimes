"""Audit frozen splits, saved models, prediction metrics and delivery artifacts."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from battery_smetimes.forecast import type_key
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data, metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    config = read(run / "config.json")
    data = Path(config["data_folder"])
    assert sha(data / "manifest.json") == config["data_manifest_sha256"]
    assert sha(config["integrity_file"]) == config["integrity_sha256"]
    manifest, cells = load_data(data, config["integrity_file"])
    splits = read(run / "splits.json")
    assert len(splits) == 10 and len(cells) == 193
    checked, rows = [], []
    for split in splits:
        folder = run / split["name"]
        train, val, test = [set(split[p]) for p in ("train", "validation", "test")]
        assert not (train & val or train & test or val & test)
        assert train | val | test == {k for k, c in cells.items() if c["metadata"]["dataset"] == split["dataset"]}
        train_types = {type_key(cells[k]["metadata"]) for k in train}
        assert all(type_key(cells[k]["metadata"]) in train_types for k in val | test)
        for leaf in (4, 16):
            model = joblib.load(folder / f"tree_leaf{leaf}.joblib")
            assert set(model["train_ids"]) == train
        result = read(folder / "result.json")
        assert result["selected"] == min(result["validation_scores"], key=result["validation_scores"].get)
        model = joblib.load(folder / "selected_model.joblib")
        assert set(model["train_ids"]) == train | val
        assert model["dataset"] == split["dataset"]
        assert (folder / "future_soh_curves.png").stat().st_size > 1000
        report = read(folder / "test/metrics.json")
        with np.load(folder / "test/predictions.npz") as arrays:
            for row in report["by_cell"]:
                key, origin = row["cell_id"], row["origin"]
                assert key in test and origin in (20, 50, 100)
                prefix = f"{key}__{origin}__"
                fc, truth, good = [arrays[prefix + field] for field in ("cycle", "truth", "regular_target")]
                assert np.all(fc > origin)
                for method in ("tree", "type_mean"):
                    calculated = metrics(truth[good], arrays[prefix + method][good])["mape_percent"]
                    assert abs(calculated - row["metrics"][method]["mape_percent"]) < 1e-10
                rows.append(row)
        checked.append({"fold": split["name"], "train": len(train), "validation": len(val),
            "test": len(test), "selected": result["selected"], "scored_origins": len(report["by_cell"])})
    replays = read(run / "reload_verification.json")
    assert replays["verified_origins"] == len(rows)
    assert max(r["max_absolute_prediction_difference"] for r in replays["checks"]) < 1e-10
    assert len(read(run / "cli_verification.json")["checks"]) == 4
    assert len(read(run / "raw_input_verification.json")["checked"]) == 6
    quality = read(data / "quality_audit.json")
    assert not ({"b2c7", "b2c8", "b2c9", "b2c15", "b2c16", "b3c37"} & set(cells))
    exclusions = read(run / "target_exclusions.json")
    assert sum(map(len, exclusions["by_cell"].values())) == 24
    summary = read(run / "summary.json")
    difficult = {}
    for domain, domain_summary in summary["domains"].items():
        stats = domain_summary["100"]
        values = stats["by_cell"]
        assert abs(np.mean(list(values.values())) - stats["cell_macro_mape_percent"]) < 1e-10
        difficult[domain] = [{"cell_id": k, "mean_mape_percent": v} for k, v in sorted(values.items(), key=lambda p: -p[1])[:5]]
    save_json(run / "hard_cases.json", difficult)
    save_json(run / "delivery_verification.json", {"passed": True, "prepared_cells": len(cells),
        "folds": checked, "metrics_recomputed_origins": len(rows),
        "raw_prefix_examples_verified": 6, "cli_saved_prediction_paths_verified": 4,
        "known_exclusions": quality["known_cell_exclusions"],
        "note": "This verifies the development delivery and split/metric artifacts; it is not a new blind accuracy evaluation."})
    print("VERIFIED", len(splits), "whole-cell folds;", len(rows), "metric rows;", len(cells), "data fingerprints")


if __name__ == "__main__":
    main()
