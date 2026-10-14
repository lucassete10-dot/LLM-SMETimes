"""Cell-first primary tables. Never present partial runs as full CV evidence."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.prepare_same_type_forecast import save_json, sha


METRICS = ("rmse_soh", "mae_soh", "mape_percent", "r2")


def describe(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return {"n_defined": len(finite), "mean": float(np.mean(finite)) if finite else None,
            "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None}


def aggregate_scores(scores, expected, seeds):
    required = {(seed, cell) for seed in seeds for cell in expected}
    if not required or set(scores) != required:
        raise ValueError("aggregation requires exact complete eligible cell/seed coverage")
    for (seed, cell), score in scores.items():
        if score["fold"] != expected[cell]:
            raise ValueError("score fold does not match frozen test membership")
        for metric in METRICS:
            value = score[metric]
            if value is None and metric == "r2":
                continue
            if value is None or not np.isfinite(value) or (metric != "r2" and value < 0):
                raise ValueError("invalid regression metric cannot be dropped from the cell denominator")
    per_cell = [{"cell_id": cell, "fold": expected[cell],
        **{metric: describe([scores[(seed, cell)][metric] for seed in seeds]) for metric in METRICS}}
        for cell in sorted(expected)]
    result = {"by_cell_training_seed_variation": per_cell, "metrics": {}}
    for metric in METRICS:
        seed_macro = [describe([scores[(seed, cell)][metric] for cell in expected])["mean"] for seed in seeds]
        fold_macro = [describe([scores[(seed, cell)][metric] for seed in seeds for cell, own_fold in expected.items()
                               if own_fold == fold])["mean"] for fold in sorted(set(expected.values()))]
        result["metrics"][metric] = {
            "seed_macro_mean_and_sd": describe(seed_macro),
            "fold_macro_descriptive_mean_and_sd_not_independent_CI": describe(fold_macro),
            "between_cell_mean_and_sd_of_seed_averaged_errors": describe([r[metric]["mean"] for r in per_cell]),
            "seed_macro_values": seed_macro, "fold_macro_values": fold_macro}
    mape = [r["mape_percent"]["mean"] for r in per_cell]
    result["fraction_cells_seed_averaged_mape_le_6"] = float(np.mean(np.asarray(mape) <= 6))
    result["worst_cell"] = max(per_cell, key=lambda r: r["mape_percent"]["mean"])
    return result


def report_run(folder, origin=100, horizon="all_observed_future", eligible_cells=None):
    config = json.loads((folder / "config.json").read_text())
    if config["smoke"]:
        return []  # no two-epoch/single-fold numbers in the primary table
    plans = json.loads((folder / "frozen_audit/splits.json").read_text())["plans"]
    frozen = json.loads((folder / "frozen_audit/config.json").read_text())["config"]
    seeds = frozen["training_seeds"]
    expected = {}
    for plan in plans:
        dataset = plan["name"].split("_fold")[0]
        members = expected.setdefault(dataset, {})
        for key in plan["test"]:
            if key in members:
                raise ValueError("test battery occurs in more than one frozen fold")
            members[key] = plan["name"]
    full_expected = {dataset: dict(members) for dataset, members in expected.items()}
    cohort_sizes = {dataset: len(members) for dataset, members in expected.items()}
    if eligible_cells is not None:
        expected = {dataset: {key: fold for key, fold in members.items() if key in eligible_cells.get(dataset, set())}
                    for dataset, members in expected.items()}
    groups, sources = {}, {}
    for path in sorted(folder.glob("*/*/*/test/metrics.json")):
        if not (path.parent.parent / "resources.json").is_file():
            continue  # the writer has not finalized this model's outputs
        fold, seed, model = path.relative_to(folder).parts[:3]
        dataset = fold.split("_fold")[0]
        group = groups.setdefault((dataset, model), {})
        sources.setdefault((dataset, model), []).append({"path": str(path.resolve()), "sha256": sha(path)})
        for row in json.loads(path.read_text())["by_cell"]:
            if row["origin"] != origin:
                continue
            if full_expected.get(dataset, {}).get(row["cell_id"]) != fold or int(seed) not in seeds:
                raise ValueError("test membership does not match the frozen primary folds")
            if row["cell_id"] not in expected[dataset]:
                continue  # eligibility is frozen independently, not selected from this model's score
            if horizon == "all_observed_future":
                measurement = row["metrics"]
            elif horizon == "late_half":
                measurement = row["late_half"]
            elif horizon in ("within_10", "within_50", "within_100"):
                measurement = row["within_horizon"].get(horizon.split("_")[1])
            else:
                raise ValueError("unknown horizon")
            if measurement is None:
                continue
            key = (int(seed), row["cell_id"])
            if key in group:
                raise ValueError("duplicate test cell within seed/model")
            group[key] = {"fold": fold, **measurement}
    result = []
    for (dataset, model), scores in sorted(groups.items()):
        required = {(seed, cell) for seed in seeds for cell in expected[dataset]}
        missing = sorted(required - set(scores))
        entry = {"run_id": folder.name, "dataset": dataset, "model": model,
            "metric_sources": sources[(dataset, model)],
            "origin_source_cycle": origin, "horizon": horizon,
            "complete_model_dataset_primary": bool(required) and not missing and not (folder / "failure.json").exists(),
            "full_cohort_cells": cohort_sizes[dataset],
            "expected_cells": len(expected[dataset]), "training_seeds": seeds,
            "completed_cell_seed_pairs": len(scores), "expected_cell_seed_pairs": len(required),
            "missing_cell_seed_pairs": missing,
            "evidence_level": "unverified four-cell NASA cohort supplement" if dataset == "NASA" else "public development grouped CV"}
        if not entry["complete_model_dataset_primary"]:
            entry["unavailable_reason"] = "no eligible cells" if not required else "failed run requires review" if (folder / "failure.json").exists() else "incomplete coverage"
            result.append(entry)
            continue
        entry.update(aggregate_scores(scores, expected[dataset], seeds))
        result.append(entry)
    return result


def render_primary_table(rows):
    """Display the existing cell-first statistics; never rescore or pick winners."""
    def format_metric(row, metric, scale=1):
        score = row["metrics"][metric]["seed_macro_mean_and_sd"]
        if score["mean"] is None:
            if metric != "r2":
                raise ValueError("a required error metric cannot be unavailable")
            return "不适用"
        if not np.isfinite(score["mean"]) or (score["sd"] is not None and not np.isfinite(score["sd"])):
            raise ValueError("nonfinite statistics cannot be displayed as valid results")
        mean = f'{scale * score["mean"]:.3f}'
        return f'{mean} ± {scale * score["sd"]:.3f}' if score["sd"] is not None else f"{mean}（SD不适用）"

    lines = ["# 未来 SOH 主结果（源循环100，完整可观察未来）", "",
        "仅完整整电芯外折和五种子覆盖的模型进入数字表。先逐电芯计算指标，再在每个种子内对电芯等权宏平均；±为五个训练种子宏平均值的描述性样本标准差，不是置信区间，也不是电芯或折的标准差。",
        "",
        "SOH计算口径为0–1；为便于阅读，MAE/RMSE在本表乘100，单位为SOH百分点（pp），不是相对误差百分数。MAPE仍为相对误差%。R²不作百分比换算，允许负值；未定义值保留为不适用，不补零。确定性模型的五次相同结果不增加独立证据。NASA四芯物理未核验、仅作补充；公开开发验证不等于盲测。",
        "",
        "| 数据集 | 模型 | 可评/全队列电芯 | MAE（pp） | RMSE（pp） | MAPE（%） | R² | R²可定义电芯 | 逐芯种子均值MAPE≤6%的比例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    complete = [row for row in rows if row["complete_model_dataset_primary"]]
    for row in complete:
        r2_cells = sum(cell["r2"]["n_defined"] > 0 for cell in row["by_cell_training_seed_variation"])
        values = [format_metric(row, "mae_soh", 100), format_metric(row, "rmse_soh", 100),
                  format_metric(row, "mape_percent"), format_metric(row, "r2")]
        lines.append(f'| {row["dataset"]} | {row["model"]} | {row["expected_cells"]}/{row["full_cohort_cells"]} | '
                     + " | ".join(values)
                     + f' | {r2_cells}/{row["expected_cells"]} | {100 * row["fraction_cells_seed_averaged_mape_le_6"]:.1f}% |')
    lines.extend(["", "## 逐芯差异与最差例（不按测试误差排除）", "",
        "此处最差例只是描述已冻结测试结果，不能用于删芯、选超参或模型路由。SD仍为该电芯五个训练种子的描述性标准差。", "",
        "| 数据集 | 模型 | 最差电芯 | 该电芯MAPE（%） |",
        "|---|---|---|---:|"])
    for row in complete:
        worst = row["worst_cell"]
        score = worst["mape_percent"]
        sd = f'{score["sd"]:.3f}' if score["sd"] is not None else "不适用"
        lines.append(f'| {row["dataset"]} | {row["model"]} | {worst["cell_id"]} | {score["mean"]:.3f} ± {sd} |')
    lines.extend(["", "逐电芯四指标、各指标可定义数量、种子宏平均原值、折间及电芯间差异均保留在同目录primary_report.json；本表仅转换显示单位，未改变原数值。未完整覆盖的模型只有进度，不参与主比较。不同未来起点和目标窗口另见既有horizon_report.json及配对早期表，不把整体低误差解释成末期准确或新数据盲测保证。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path)
    args = parser.parse_args()
    runs = args.run or sorted(Path("experiments/soh_research_v5/runs").glob("*"))
    rows = []
    for run in runs:
        if (run / "config.json").exists() and (run / "progress.json").exists():
            rows.extend(report_run(run))
    folder = Path("results/soh_research_v5_primary") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, folder / "report_source.py")
    save_json(folder / "primary_report.json", {"primary": rows,
        "reporting_source_sha256": sha(Path(__file__)), "numpy_version": np.__version__,
        "contract": "Cells are independent evaluation units. Seeds and overlapping folds are distinct descriptive variability levels, not extra batteries. Incomplete model/dataset coverage has no primary score. Constant-curve R2 retains missing values. NASA remains supplemental. Deterministic models' identical seed repeats are not independent stochastic evidence."})
    (folder / "primary_table.md").write_text(render_primary_table(rows), encoding="utf-8")
    save_json(Path("results/latest_soh_research_v5_primary.json"), {"folder": str(folder.resolve())})
    print(json.dumps({"folder": str(folder.resolve()), "complete_model_dataset_rows": sum(r["complete_model_dataset_primary"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
