"""Restore only an absent training diagnostic from existing saved predictions.

No fitting or metric/model changes. The original frozen drawing function is
used; this is an engineering diagnostic, not a publication-validated figure.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import shutil

import numpy as np

from scripts.audit_soh_research_artifacts_v5 import audit_scores, fingerprint, read_json
from scripts.prepare_same_type_forecast import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", type=Path)
    args = parser.parse_args()
    model = args.model_folder.resolve()
    run = model.parents[2]
    if not run.is_relative_to(Path("experiments/soh_research_v5/runs").resolve()):
        raise ValueError("repair target is outside the explicit research runs")
    target = model / "test/curves_origin100.png"
    if target.exists():
        raise ValueError("existing diagnostic will not be overwritten")
    if not (model / "resources.json").is_file():
        raise ValueError("unit is not finalized; let its training worker finish")
    config, unit = read_json(run / "config.json"), read_json(model / "config.json")
    if unit["plan"] not in read_json(run / "frozen_audit/splits.json")["plans"]:
        raise ValueError("unit has no matching frozen split")
    report = audit_scores(model / "test", unit["plan"]["test"], config["config"]["origins_source_cycles"], unit["physical_family"])
    driver = run / "source_snapshot/train_soh_research_v5.py"
    expected_driver = next(value for path, value in config["source_sha256"].items() if Path(path).name == driver.name)
    if fingerprint(driver) != expected_driver:
        raise ValueError("frozen original plotting driver changed")
    before = {str(path.resolve()): fingerprint(path) for path in (model / "test/metrics.json", model / "test/predictions.npz")}
    output = Path("experiments/soh_research_v5/artifact_repairs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "repair_source.py")
    shutil.copy2(driver, output / "original_plot_source.py")
    save_json(output / "config.json", {"model_folder": str(model), "original_scores_sha256": before,
        "original_driver_sha256": expected_driver, "repair_source_sha256": fingerprint(Path(__file__)),
        "policy": "Only fill a confirmed absent PNG from existing predictions; never fit, change weights, replace metrics or overwrite existing figures."})
    spec = importlib.util.spec_from_file_location("original_plot_driver", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(model / "test/predictions.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    module.save_scores(output / "diagnostic", report, arrays)
    figure = output / "diagnostic/curves_origin100.png"
    if not figure.is_file() or target.exists():
        raise ValueError("no source-N100 diagnostic produced or target became occupied")
    if any(fingerprint(path) != digest for path, digest in before.items()):
        raise ValueError("original scores changed during read-only repair")
    shutil.copy2(figure, target)
    save_json(output / "report.json", {"status": "missing diagnostic restored without fitting or metric changes",
        "target": str(target), "figure_sha256": fingerprint(target), "original_scores_unchanged": True,
        "scope": "Original engineering PNG only; no publication layout/collision validation is claimed."})
    print(str(output.resolve()))


if __name__ == "__main__":
    main()
