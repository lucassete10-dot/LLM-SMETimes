"""Frozen whole-cell same-type development forecast experiments."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import joblib
import numpy as np

from battery_smetimes.forecast import DOMAINS, ORIGINS, SEED, fit_curve_model, predict_curve, type_key
from scripts.prepare_same_type_forecast import ROOT as DATA, save_json, sha

OUT = Path("battery_models_v4/same_type_forecast")


def load_data(folder, integrity=None):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    cells = {}
    for record in manifest["cells"]:
        if sha(record["file"]) != record["sha256"]:
            raise ValueError("prepared data fingerprint changed")
        with np.load(record["file"]) as value:
            cells[record["cell_id"]] = {"metadata": record["metadata"],
                "cycle": value["cycle"], "raw": value["raw_features"],
                "label_cycle": value["label_cycle"], "soh": value["soh"]}
    exclusions = defaultdict(set)
    if integrity is not None:
        for row in json.loads(Path(integrity).read_text(encoding="utf-8"))["issues"]:
            exclusions[row["cell_id"]].add(row["cycle"])
    for key, cell in cells.items():
        cell["unfiltered_soh"] = cell["soh"].copy()
        rejected = np.isin(cell["label_cycle"], list(exclusions[key]))
        cell["soh"][rejected] = np.nan
        valid_input = ~np.isin(cell["cycle"], list(exclusions[key]))
        cell["excluded_input_cycles"] = cell["cycle"][~valid_input].tolist()
        cell["cycle"], cell["raw"] = cell["cycle"][valid_input], cell["raw"][valid_input]
    return manifest, cells


def make_splits(cells):
    plans = []
    for domain in DOMAINS:
        groups = defaultdict(list)
        for key, value in sorted(cells.items()):
            if value["metadata"]["dataset"] == domain:
                groups[type_key(value["metadata"])].append(key)
        if domain == "NASA":
            keys = sorted(key for group in groups.values() for key in group)
            for j, key in enumerate(keys):
                val = keys[(j + 1) % len(keys)]
                plans.append({"name": f"NASA_fold{j + 1}", "dataset": domain, "seed": SEED,
                    "train": [k for k in keys if k not in (key, val)], "validation": [val], "test": [key]})
            continue
        for repeat in range(3):
            rng = np.random.default_rng(SEED + repeat)
            split = {part: [] for part in ("train", "validation", "test")}
            for group, keys in sorted(groups.items()):
                shuffled = rng.permutation(keys).tolist()
                if len(keys) < 3:
                    raise ValueError(f"{group}: insufficient cells for train/validation/test")
                n_test = max(1, int(np.ceil(.2 * len(keys))))
                n_val = max(1, int(np.ceil(.2 * (len(keys) - n_test))))
                split["test"].extend(shuffled[:n_test])
                split["validation"].extend(shuffled[n_test:n_test + n_val])
                split["train"].extend(shuffled[n_test + n_val:])
            plans.append({"name": f"{domain}_repeat{repeat + 1}", "dataset": domain,
                          "seed": SEED + repeat, **split})
    for plan in plans:
        parts = [set(plan[p]) for p in ("train", "validation", "test")]
        if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("cell split overlap")
        types = {type_key(cells[k]["metadata"]) for k in plan["train"]}
        if any(type_key(cells[k]["metadata"]) not in types for k in plan["test"] + plan["validation"]):
            raise ValueError("held-out type absent from training")
    return plans


def metrics(truth, prediction):
    return {"mape_percent": float(100 * np.mean(np.abs(prediction - truth) / truth)),
            "mae_soh": float(np.mean(np.abs(prediction - truth)))}


def save_evaluation_plot(folder, report, arrays, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for row in report["by_cell"] if row["origin"] == 100]
    if not rows:
        return
    columns = min(4, len(rows))
    fig, axes = plt.subplots(int(np.ceil(len(rows) / columns)), columns,
        figsize=(4.5 * columns, 2.8 * int(np.ceil(len(rows) / columns))), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, row in zip(axes.flat, rows):
        ax.set_visible(True)
        prefix = row["cell_id"] + "__100__"
        cycle = arrays[prefix + "cycle"]
        ax.plot(cycle, arrays[prefix + "truth"], color="black", label="Observed SOH")
        for method in methods:
            ax.plot(cycle, arrays[prefix + method], label=method, lw=1)
        ax.set(title=row["cell_id"], xlabel="Future source cycle", ylabel="SOH")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "curves_origin100.png", dpi=120)
    plt.close(fig)


def evaluate(bundle, cells, ids, folder, methods):
    rows, skipped = [], []
    arrays = {}
    for key in ids:
        cell = cells[key]
        for origin in ORIGINS:
            original = cell.get("unfiltered_soh", cell["soh"])
            future = np.isfinite(original) & (original > 0) & (cell["label_cycle"] > origin)
            fc, truth = cell["label_cycle"][future], original[future]
            valid_target = np.isfinite(cell["soh"][future])
            if valid_target.sum() < 2:
                skipped.append({"cell_id": key, "origin": origin, "reason": "less than two observed future targets"})
                continue
            curves = {}
            try:
                for method in methods:
                    curves[method], support = predict_curve(bundle, cell["cycle"], cell["raw"],
                        cell["metadata"], origin, fc, method)
            except ValueError as error:
                if "insufficient" in str(error):
                    skipped.append({"cell_id": key, "origin": origin, "reason": str(error)})
                    continue
                raise
            own_type = type_key(cell["metadata"])
            peers = [k for k in bundle["train_ids"] if type_key(cells[k]["metadata"]) == own_type]
            if cell["metadata"]["dataset"] == "NASA":
                exact_seen = any(cells[k]["metadata"]["discharge_cutoff_v"] == cell["metadata"]["discharge_cutoff_v"] for k in peers)
            else:
                exact_seen = any(cells[k]["metadata"]["charge_protocol"] == cell["metadata"]["charge_protocol"] for k in peers)
            row = {"cell_id": key, "type": own_type, "origin": origin, "future_points": len(fc),
                "regular_future_points": int(valid_target.sum()), "interrupted_target_points_excluded": int((~valid_target).sum()),
                "last_observed_cycle": int(fc[-1]), "same_type_training_cells": len(peers),
                "exact_protocol_seen": exact_seen, "future_points_beyond_all_same_type_training_records": int((support == 0).sum()),
                "metrics": {}}
            for method, prediction in curves.items():
                scores = metrics(truth[valid_target], prediction[valid_target])
                scores["unfiltered_mape_percent"] = metrics(truth, prediction)["mape_percent"]
                tail = valid_target & (fc >= (fc[0] + fc[-1]) / 2)
                scores["late_half_mape_percent"] = metrics(truth[tail], prediction[tail])["mape_percent"] if tail.any() else None
                scores["within_horizon_mape_percent"] = {str(h): metrics(truth[(fc <= origin + h) & valid_target], prediction[(fc <= origin + h) & valid_target])["mape_percent"]
                    if np.any((fc <= origin + h) & valid_target) else None for h in (10, 50, 100)}
                row["metrics"][method] = scores
                arrays[f"{key}__{origin}__{method}"] = prediction
            arrays[f"{key}__{origin}__cycle"] = fc
            arrays[f"{key}__{origin}__truth"] = truth
            arrays[f"{key}__{origin}__regular_target"] = valid_target
            arrays[f"{key}__{origin}__same_type_support"] = support
            rows.append(row)
    summary = {}
    for origin in ORIGINS:
        chosen = [r for r in rows if r["origin"] == origin]
        summary[str(origin)] = {method: {
            "cells": len(chosen), "cell_macro_mape_percent": float(np.mean([r["metrics"][method]["mape_percent"] for r in chosen])) if chosen else None,
            "fraction_cells_mape_le_6": float(np.mean([r["metrics"][method]["mape_percent"] <= 6 for r in chosen])) if chosen else None}
            for method in methods}
    report = {"summary": summary, "by_cell": rows, "skipped": skipped}
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions.npz", **arrays)
    save_json(folder / "metrics.json", report)
    save_evaluation_plot(folder, report, arrays, methods)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--integrity", type=Path)
    args = parser.parse_args()
    data = args.data or Path(json.loads((DATA / "latest.json").read_text())["folder"])
    integrity = args.integrity or Path(json.loads((DATA / "latest_integrity.json").read_text())["file"])
    manifest, cells = load_data(data, integrity)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / "runs" / stamp
    run.mkdir(parents=True, exist_ok=False)
    splits = make_splits(cells)
    save_json(run / "config.json", {"data_folder": str(data.resolve()), "data_manifest_sha256": sha(data / "manifest.json"),
        "integrity_file": str(integrity.resolve()), "integrity_sha256": sha(integrity),
        "selection": "minimum validation cell-macro full-future MAPE from origin 100 only",
        "model_candidates": ["type_mean", "tree_leaf4", "tree_leaf16"],
        "seed": SEED, "origins_source_cycles": ORIGINS,
        "code_sha256": {p: sha(p) for p in [__file__, "battery_smetimes/forecast.py", "scripts/prepare_same_type_forecast.py"]},
        "future_assumption": "same known type/operating strategy continues",
        "status": "already-inspected public-data development; not blind acceptance"})
    save_json(run / "splits.json", splits)
    save_json(run / "target_exclusions.json", {"rule_source": str(integrity.resolve()), "by_cell": {
        key: cell["label_cycle"][np.isfinite(cell["unfiltered_soh"]) & ~np.isfinite(cell["soh"])].tolist()
        for key, cell in cells.items() if np.any(np.isfinite(cell["unfiltered_soh"]) & ~np.isfinite(cell["soh"]))},
        "signal_history_exclusions": {key: cell["excluded_input_cycles"] for key, cell in cells.items() if cell["excluded_input_cycles"]}})
    save_json(run / "classification.json", {"type_counts": manifest["type_counts"],
        "by_cell": [{"cell_id": r["cell_id"], "type": r["type"], "metadata": r["metadata"]} for r in manifest["cells"]]})
    print("RUN", run.resolve(), flush=True)
    completed = []
    try:
        for split in splits:
            folder = run / split["name"]
            folder.mkdir()
            save_json(folder / "split.json", split)
            scores = {}
            for leaf in (4, 16):
                name = f"tree_leaf{leaf}"
                print("FIT", split["name"], name, flush=True)
                bundle = fit_curve_model(cells, split["train"], leaf, seed=split["seed"])
                joblib.dump(bundle, folder / f"{name}.joblib", compress=3)
                report = evaluate(bundle, cells, split["validation"], folder / name, ("tree", "type_mean"))
                scores[name] = report["summary"]["100"]["tree"]["cell_macro_mape_percent"]
                scores["type_mean"] = report["summary"]["100"]["type_mean"]["cell_macro_mape_percent"]
                save_json(folder / "validation_scores.json", scores)
            if any(v is None for v in scores.values()):
                raise ValueError("missing validation origin 100 coverage")
            selected = min(scores, key=scores.get)
            leaf = int(selected.replace("tree_leaf", "")) if selected.startswith("tree") else 4
            # Refit on all development cells after validation-only model selection.
            bundle = fit_curve_model(cells, split["train"] + split["validation"], leaf, seed=split["seed"])
            bundle["selected_method"] = "type_mean" if selected == "type_mean" else "tree"
            bundle["selected_name"] = selected
            bundle["dataset"] = split["dataset"]
            bundle["input_contract"] = "raw_features extracted from observed V/I/T prefixes; capacity/SOH/EOL forbidden as input"
            joblib.dump(bundle, folder / "selected_model.joblib", compress=3)
            report = evaluate(bundle, cells, split["test"], folder / "test", ("tree", "type_mean"))
            final = {"name": split["name"], "dataset": split["dataset"], "selected": selected,
                "selected_method": bundle["selected_method"], "validation_scores": scores,
                "test_summary": report["summary"], "model_file": str((folder / "selected_model.joblib").resolve())}
            save_json(folder / "result.json", final)
            completed.append(final)
            save_json(run / "progress.json", {"completed": len(completed), "expected": len(splits)})
            print("SAVED", split["name"], selected, "test MAPE", report["summary"]["100"][bundle["selected_method"]], flush=True)
        save_json(run / "results.json", {"runs": completed})
        save_json(OUT / "latest.json", {"run": str(run.resolve())})
        print("COMPLETE", run.resolve(), flush=True)
    except BaseException:
        save_json(run / "failure.json", {"traceback": traceback.format_exc(), "completed": len(completed)})
        raise


if __name__ == "__main__":
    main()
