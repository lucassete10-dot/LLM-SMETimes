"""Run fixed physical-family baseline experiments, saving every result immediately.

First use --smoke. Non-smoke defaults to all 14 frozen folds and five seeds;
these two baselines alone are NOT the completed revised research goal.
"""

import argparse
from datetime import datetime, timezone
import json
from importlib.metadata import version
import logging
import os
from pathlib import Path
import platform
import shutil
import time
import traceback

import joblib
import numpy as np
import sklearn

from battery_smetimes.physical_types import load_registry
from battery_smetimes.research_baselines import evaluate_baseline, fit_baseline
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data
from scripts.summarize_soh_research_v5 import summarize


def save_scores(folder, report, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "metrics.json", report)
    np.savez_compressed(folder / "predictions.npz", **arrays)
    rows = [r for r in report["by_cell"] if r["origin"] == 100]
    if not rows:
        return
    columns = min(4, len(rows))
    fig, axes = plt.subplots(int(np.ceil(len(rows) / columns)), columns,
        figsize=(4.4 * columns, 2.8 * int(np.ceil(len(rows) / columns))), squeeze=False)
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, row in zip(axes.flat, rows):
        axis.set_visible(True)
        prefix = f'{row["cell_id"]}__100__'
        axis.plot(arrays[prefix + "cycle"], arrays[prefix + "truth"], color="black", label="Observed")
        axis.plot(arrays[prefix + "cycle"], arrays[prefix + "prediction"], label="Forecast")
        axis.set(title=f'{row["cell_id"]}: {row["metrics"]["mape_percent"]:.2f}%',
                 xlabel="Future source cycle", ylabel="SOH")
        axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "curves_origin100.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-folder", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold", action="append", help="Select a frozen fold; do not change membership")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--model-config", type=Path, help="Additional preset model candidates; frozen data/folds stay unchanged")
    parser.add_argument("--model", action="append", help="Restrict to named presets in the selected configuration")
    parser.add_argument("--no-global-summary", action="store_true", help="Avoid concurrent writers; another active v5 run will still ingest these saved metrics")
    args = parser.parse_args()
    audit_folder = args.audit_folder or Path(json.loads(Path("experiments/soh_research_v5/latest_audit.json").read_text())["folder"])
    frozen = json.loads((audit_folder / "config.json").read_text())
    config = frozen["config"]
    extra = json.loads(args.model_config.read_text(encoding="utf-8")) if args.model_config else None
    if extra is not None and Path(extra["frozen_plan"]).resolve() != audit_folder.resolve():
        raise ValueError("additional baseline configuration refers to a different frozen plan")
    model_candidates = extra["models"] if extra is not None else config["initial_candidates"]
    if args.model:
        if set(args.model) - set(model_candidates):
            raise ValueError("unknown configured model")
        model_candidates = {key: value for key, value in model_candidates.items() if key in args.model}
    audit = json.loads((audit_folder / "audit.json").read_text())
    registered = json.loads((audit_folder / "cell_registry.json").read_text())["cells"]
    registry_file = audit_folder / "source_snapshot/battery_physical_registry.json"
    if sha(registry_file) != frozen["registry_sha256"] or sha(config["integrity_audit"]) != frozen["integrity_sha256"]:
        raise ValueError("frozen physical registry or integrity rules changed")
    registry = load_registry(registry_file)
    data = Path(audit["prepared_folder"])
    if sha(data / "manifest.json") != frozen["prepared_manifest_sha256"]:
        raise ValueError("frozen prepared manifest changed")
    manifest, cells = load_data(data, config["integrity_audit"])
    for key, cell in cells.items():
        cell.update({name: registered[key][name] for name in (
            "physical_family", "reference_established_source_cycle")})
    plans = json.loads((audit_folder / "splits.json").read_text())["plans"]
    fold_names = args.fold or (config["smoke"]["folds"] if args.smoke else [r["name"] for r in plans])
    if set(fold_names) - {r["name"] for r in plans}:
        raise ValueError("unknown frozen fold")
    plans = [r for r in plans if r["name"] in fold_names]
    seeds = args.seed or ([config["smoke"]["training_seed"]] if args.smoke else config["training_seeds"])
    if set(seeds) - set(config["training_seeds"]):
        raise ValueError("seed outside preregistered training seeds")
    folder = Path("experiments/soh_research_v5/runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(folder / "training.log", encoding="utf-8"), logging.StreamHandler()])
    source_files = [Path(__file__), Path("battery_smetimes/research_baselines.py"),
        Path("battery_smetimes/research_splits.py"), Path("battery_smetimes/physical_types.py"),
        Path("battery_smetimes/forecast.py"), Path("scripts/train_same_type_forecast.py"),
        Path("scripts/prepare_same_type_forecast.py"), Path("scripts/summarize_soh_research_v5.py")]
    if args.model_config:
        source_files.extend([args.model_config, Path("battery_smetimes/research_extended_baselines.py")])
    snapshot = folder / "source_snapshot"
    source_hashes = {}
    for file in source_files:
        target = snapshot / file.as_posix().replace(":", "") if not file.is_absolute() else snapshot / file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        source_hashes[str(file)] = sha(file)
    shutil.copytree(audit_folder, folder / "frozen_audit")
    save_json(folder / "config.json", {"config": config, "folds": fold_names, "seeds": seeds,
        "smoke": args.smoke, "tree_estimators": 48 if args.smoke else 160,
        "selection_metric": "validation cell-macro full-future MAPE at origin 100",
        "source_sha256": source_hashes, "audit_manifest": str(audit_folder),
        "model_candidates": model_candidates, "additional_baseline_config": extra,
        "prepared_files_sha256": {r["cell_id"]: r["sha256"] for r in manifest["cells"]},
        "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
        "packages": {key: version(key) for key in ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib")},
        "optional_packages": {key: version(key) for key in ("lightgbm", "psutil")} if extra is not None else {},
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "evidence_level": "single-fold single-seed engineering smoke" if args.smoke else "public development grouped CV"})
    save_json(folder / "splits.json", {"plans": plans})
    summary = {"complete": False, "smoke": args.smoke, "completed": [], "summary": [],
        "pending_models_and_ablations": config["full_models_pending"] + config["ablations_pending"]}
    save_json(folder / "progress.json", summary)
    save_json(Path("experiments/soh_research_v5/latest_run.json"), {"folder": str(folder.resolve())})
    try:
        for plan in plans:
            family = registered[plan["train"][0]]["physical_family"]
            for seed in seeds:
                for model_name, candidates in model_candidates.items():
                    destination = folder / plan["name"] / str(seed) / model_name
                    destination.mkdir(parents=True, exist_ok=False)
                    save_json(destination / "config.json", {"plan": plan, "seed": seed, "model": model_name,
                        "candidates": candidates, "physical_family": family, "smoke": args.smoke})
                    scores = []
                    for index, parameters in enumerate(candidates):
                        logging.info("FIT %s seed=%s %s candidate=%s", plan["name"], seed, model_name, index)
                        candidate_dir = destination / f"candidate_{index}"
                        candidate_dir.mkdir()
                        bundle = fit_baseline(cells, plan["train"], family, registry, config,
                                              model_name, parameters, seed, args.smoke)
                        joblib.dump(bundle, candidate_dir / "model.joblib")
                        report, arrays = evaluate_baseline(bundle, cells, plan["validation"])
                        save_scores(candidate_dir / "validation", report, arrays)
                        score = report["summary"]["100"]["cell_macro_mape_percent"]
                        if score is None:
                            raise ValueError("no validation cells at the primary origin")
                        scores.append({"index": index, "parameters": parameters,
                                       "validation_mape_percent": score, "fit_seconds": bundle["fit_seconds"],
                                       "fit_warnings": bundle.get("fit_warnings", [])})
                        save_json(destination / "validation_scores.json", scores)
                    selected = min(scores, key=lambda r: r["validation_mape_percent"])
                    save_json(destination / "selected.json", selected)
                    # Only after validation selection do we refit and score test.
                    bundle = fit_baseline(cells, plan["train"] + plan["validation"], family, registry, config,
                                          model_name, selected["parameters"], seed, args.smoke)
                    if set(bundle["train_ids"]) & set(plan["test"]):
                        raise ValueError("test cells entered fitting")
                    joblib.dump(bundle, destination / "model.joblib")
                    started = time.perf_counter()
                    report, arrays = evaluate_baseline(bundle, cells, plan["test"])
                    save_scores(destination / "test", report, arrays)
                    process_memory = None
                    try:
                        import psutil
                        process_memory = psutil.Process().memory_info().rss / 2 ** 20
                    except ImportError:
                        pass
                    save_json(destination / "resources.json", {"fit_seconds": bundle["fit_seconds"],
                        "test_evaluation_and_plot_seconds": time.perf_counter() - started,
                        "process_rss_mb_at_end_not_peak": process_memory, "gpu_used": False,
                        "saved_model_bytes": (destination / "model.joblib").stat().st_size,
                        "training_pairs": bundle["training_pairs"]})
                    save_json(destination / "fit_diagnostics.json", {key: bundle.get(key) for key in (
                        "fit_warnings", "iterations", "dual_gap", "fitted_kernel", "estimator_training_pairs", "coreset_by_cell")})
                    for origin, row in report["summary"].items():
                        summary["summary"].append({"fold": plan["name"], "dataset": plan["dataset"],
                            "physical_family": family, "seed": seed, "model": model_name, "origin": int(origin), **row})
                    summary["completed"].append({"fold": plan["name"], "seed": seed, "model": model_name})
                    save_json(folder / "progress.json", summary)
                    if not args.no_global_summary:
                        summarize()
                    logging.info("SAVED %s seed=%s %s origin100 MAPE=%s", plan["name"], seed,
                                 model_name, report["summary"]["100"]["cell_macro_mape_percent"])
        summary["complete"] = True
        save_json(folder / "summary.json", summary)
        save_json(folder / "progress.json", summary)
        if not args.no_global_summary:
            summarize()
        logging.info("COMPLETE RUN %s", folder.resolve())
    except BaseException:
        save_json(folder / "failure.json", {"traceback": traceback.format_exc(), "completed": summary["completed"]})
        raise


if __name__ == "__main__":
    main()
