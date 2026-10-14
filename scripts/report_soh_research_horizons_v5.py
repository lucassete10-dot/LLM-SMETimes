"""Prespecified origin/horizon tables with independent eligibility denominators."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from battery_smetimes.research_baselines import signal_prefix, regression_metrics
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.report_soh_research_primary_v5 import report_run, aggregate_scores
from scripts.train_same_type_forecast import load_data


HORIZONS = ("all_observed_future", "within_10", "within_50", "within_100", "late_half")


def matched_future_report(run, grid):
    plans = json.loads((run / "frozen_audit/splits.json").read_text())["plans"]
    frozen = json.loads((run / "frozen_audit/config.json").read_text())["config"]
    origins, seeds = frozen["origins_source_cycles"], frozen["training_seeds"]
    datasets = {plan["name"].split("_fold")[0] for plan in plans}
    common = {dataset: set.intersection(*(grid[(origin, "all_observed_future")][dataset] for origin in origins)) for dataset in datasets}
    expected = {dataset: {cell: plan["name"] for plan in plans if plan["name"].split("_fold")[0] == dataset
                         for cell in plan["test"] if cell in common[dataset]} for dataset in datasets}
    groups, sources = {}, {}
    for path in sorted(run.glob("*/*/*/test/predictions.npz")):
        if not (path.parent.parent / "resources.json").is_file():
            continue  # do not read an NPZ while save_scores is writing it
        fold, seed, model = path.relative_to(run).parts[:3]
        dataset = fold.split("_fold")[0]
        sources.setdefault((dataset, model), []).append({"path": str(path.resolve()), "sha256": sha(path)})
        ids = [cell for cell, own_fold in expected[dataset].items() if own_fold == fold]
        with np.load(path) as arrays:
            for origin in origins:
                group = groups.setdefault((dataset, model, origin), {})
                for cell in ids:
                    prefix = f"{cell}__{origin}__"
                    if prefix + "cycle" not in arrays:
                        continue
                    reference = f"{cell}__{max(origins)}__"
                    take = arrays[prefix + "cycle"] > max(origins)
                    cycles = arrays[prefix + "cycle"][take]
                    truth = arrays[prefix + "truth"][take]
                    if not np.array_equal(cycles, arrays[reference + "cycle"]) or not np.array_equal(truth, arrays[reference + "truth"]):
                        raise ValueError("matched-prefix comparison has unequal future target grids")
                    key = (int(seed), cell)
                    if key in group or int(seed) not in seeds:
                        raise ValueError("invalid matched-prefix cell/seed membership")
                    group[key] = {"fold": fold, **regression_metrics(truth, arrays[prefix + "prediction"][take])}
    rows = []
    for (dataset, model, origin), scores in sorted(groups.items()):
        required = {(seed, cell) for seed in seeds for cell in expected[dataset]}
        missing = sorted(required - set(scores))
        complete = bool(required) and not missing and not (run / "failure.json").exists()
        entry = {"run_id": run.name, "dataset": dataset, "model": model,
            "prediction_sources": sources[(dataset, model)],
            "origin_source_cycle": origin, "horizon": f"matched_future_after_source{max(origins)}",
            "evaluation_cells": len(expected[dataset]), "complete_coverage": complete,
            "expected_cell_seed_pairs": len(required), "completed_cell_seed_pairs": len(scores),
            "missing": missing, "training_seeds": seeds,
            "evidence_level": "four-cell NASA supplement" if dataset == "NASA" else "public development grouped CV"}
        if complete:
            entry.update(aggregate_scores(scores, expected[dataset], seeds))
        rows.append(entry)
    return rows


def expected_eligibility(cells, origin, horizon):
    eligible, skipped = {}, []
    for key, cell in cells.items():
        dataset = cell["metadata"]["dataset"]
        eligible.setdefault(dataset, set())
        try:
            signal_prefix(cell, origin)
        except ValueError as error:
            if "insufficient" not in str(error) and "not established" not in str(error):
                raise
            skipped.append({"cell_id": key, "dataset": dataset, "reason": str(error)})
            continue
        targets = cell["label_cycle"][np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin)]
        if len(targets) < 2:
            skipped.append({"cell_id": key, "dataset": dataset, "reason": "insufficient valid future targets"})
            continue
        if horizon.startswith("within_") and not np.any(targets <= origin + int(horizon.split("_")[1])):
            skipped.append({"cell_id": key, "dataset": dataset, "reason": "no comparable diagnostic target in this source-cycle horizon"})
            continue
        eligible[dataset].add(key)
    return eligible, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append")
    args = parser.parse_args()
    runs = args.run or sorted(Path("experiments/soh_research_v5/runs").glob("*"))
    rows, coverage, cache, matched = [], [], {}, []
    for run in runs:
        if not (run / "config.json").is_file() or not (run / "progress.json").is_file():
            continue
        config = json.loads((run / "config.json").read_text())
        if config["smoke"]:
            continue
        frozen_dir = run / "frozen_audit"
        frozen = json.loads((frozen_dir / "config.json").read_text())
        audit = json.loads((frozen_dir / "audit.json").read_text())
        fingerprint = (frozen["prepared_manifest_sha256"], frozen["integrity_sha256"], frozen["registry_sha256"])
        if fingerprint not in cache:
            prepared = Path(audit["prepared_folder"])
            if sha(prepared / "manifest.json") != frozen["prepared_manifest_sha256"] or sha(frozen["config"]["integrity_audit"]) != frozen["integrity_sha256"]:
                raise ValueError("frozen eligibility inputs changed")
            _, cells = load_data(prepared, frozen["config"]["integrity_audit"])
            registered = json.loads((frozen_dir / "cell_registry.json").read_text())["cells"]
            for key, cell in cells.items():
                cell.update({name: registered[key][name] for name in ("physical_family", "reference_established_source_cycle")})
            grid = {}
            for origin in frozen["config"]["origins_source_cycles"]:
                for horizon in HORIZONS:
                    eligible, skipped = expected_eligibility(cells, origin, horizon)
                    grid[(origin, horizon)] = eligible
                    coverage.append({"fingerprint": fingerprint, "origin_source_cycle": origin, "horizon": horizon,
                        "eligible_ids_by_dataset": {dataset: sorted(ids) for dataset, ids in eligible.items()}, "skipped": skipped})
            cache[fingerprint] = grid
        for (origin, horizon), eligible in cache[fingerprint].items():
            rows.extend(report_run(run, origin=origin, horizon=horizon, eligible_cells=eligible))
        matched.extend(matched_future_report(run, cache[fingerprint]))
    folder = Path("results/soh_research_v5_horizons") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "horizon_report.json", {"rows": rows, "coverage": coverage, "matched_early_forecast": matched,
        "contract": "Origins are SOURCE cycles, not observed diagnostic row counts or true-life percentages. Short horizons use actual diagnostics without fabricated interpolation targets. Eligibility is derived once from audited signal/reference/target availability, not model success or test error. Only complete eligible-cell coverage at all five seeds receives numbers. Primary is origin100/all observed future; all other rows are prespecified secondary descriptive analyses. NASA is supplemental."})
    lines = ["# 早期与短程未来 SOH（预设次要评价）", "", "起点N和步长均为源循环数，不是真实寿命百分比。±是训练种子宏平均的描述性SD；完整电芯覆盖的模型才报数值。短程无可比诊断目标的电芯计入缺覆盖，不插值制造真值。NASA仅补充。", "", "| 数据集 | 模型 | N | 未来范围 | 可评/总电芯 | MAPE %（种子均值 ± SD） |", "|---|---|---:|---|---:|---:|"]
    for row in rows:
        if row["complete_model_dataset_primary"]:
            metric = row["metrics"]["mape_percent"]["seed_macro_mean_and_sd"]
            lines.append(f'| {row["dataset"]} | {row["model"]} | {row["origin_source_cycle"]} | {row["horizon"]} | {row["expected_cells"]}/{row["full_cohort_cells"]} | {metric["mean"]:.3f} ± {metric["sd"]:.3f} |')
    (folder / "horizon_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    matched_lines = ["# 同电芯、同未来目标的前缀长度对照", "", "所有N使用共同可评电芯，且仅评价源循环100之后同一组真实未来目标，避免改变电芯构成或用更多早期平坦区稀释误差。此表为描述性敏感性对照，不替代主表。", "", "| 数据集 | 模型 | N | 相同电芯数 | MAPE %（种子均值 ± SD） |", "|---|---|---:|---:|---:|"]
    for row in matched:
        if row["complete_coverage"]:
            metric = row["metrics"]["mape_percent"]["seed_macro_mean_and_sd"]
            matched_lines.append(f'| {row["dataset"]} | {row["model"]} | {row["origin_source_cycle"]} | {row["evaluation_cells"]} | {metric["mean"]:.3f} ± {metric["sd"]:.3f} |')
    (folder / "matched_early_table.md").write_text("\n".join(matched_lines) + "\n", encoding="utf-8")
    for path in (Path(__file__), Path("scripts/report_soh_research_primary_v5.py")):
        shutil.copy2(path, folder / path.name)
    save_json(Path("results/latest_soh_research_v5_horizons.json"), {"folder": str(folder.resolve())})
    print(json.dumps({"folder": str(folder.resolve()), "complete_rows": sum(r["complete_model_dataset_primary"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
