"""Render every held-out curve and replay saved models without future inputs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from battery_smetimes.forecast import DOMAINS, ORIGINS, predict_curve
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import OUT, load_data


def aggregate(rows):
    per_cell = defaultdict(list)
    for row in rows:
        per_cell[row["cell_id"]].append(row["selected_mape_percent"])
    values = [float(np.mean(v)) for v in per_cell.values()]
    def metric_average(field, horizon=None):
        grouped = defaultdict(list)
        for row in rows:
            item = row["metrics"][row["selected_method"]]
            value = item[field] if horizon is None else item[field].get(str(horizon))
            if value is not None and np.isfinite(value):
                grouped[row["cell_id"]].append(float(value))
        cell_values = [float(np.mean(group)) for group in grouped.values()]
        return {"cell_macro_mape_percent": float(np.mean(cell_values)) if cell_values else None,
                "cells_with_observations": len(cell_values)}

    return {"unique_test_cells": len(values), "test_appearances": len(rows),
        "cell_macro_mape_percent": float(np.mean(values)) if values else None,
        "fraction_cells_mape_le_6": float(np.mean(np.array(values) <= 6)) if values else None,
        "worst_cell_mean_mape_percent": float(max(values)) if values else None,
        "observed_future_span_cycles_min_median_max": np.quantile(
            [r["last_observed_cycle"] - r["origin"] for r in rows], [0, .5, 1]).tolist() if rows else None,
        "regular_future_points_min_median_max": np.quantile(
            [r["regular_future_points"] for r in rows], [0, .5, 1]).tolist() if rows else None,
        "late_half": metric_average("late_half_mape_percent"),
        "unfiltered": metric_average("unfiltered_mape_percent"),
        "within_horizon": {str(h): metric_average("within_horizon_mape_percent", h) for h in (10, 50, 100)},
        "by_cell": {k: float(np.mean(v)) for k, v in per_cell.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    run = args.run or Path(json.loads((OUT / "latest.json").read_text())["run"])
    results = json.loads((run / "results.json").read_text())["runs"]
    config = json.loads((run / "config.json").read_text())
    sources = run / "source_snapshot"
    sources.mkdir(exist_ok=True)
    saved_sources = {}
    for path in ("battery_smetimes/forecast.py", "scripts/train_same_type_forecast.py",
            "scripts/prepare_same_type_forecast.py", "scripts/audit_mit_discharge_integrity.py",
            "scripts/extract_forecast_prefix.py", "scripts/predict_same_type_forecast.py",
            "scripts/report_same_type_forecast.py", "prepare_mit.py", "prepare_xjtu.py",
            "adapt_nasa_mat_v2.py"):
        shutil.copy2(path, sources / Path(path).name)
        saved_sources[path] = sha(path)
    save_json(run / "environment.json", {"python": platform.python_version(),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__,
        "source_snapshot_sha256": saved_sources})
    _, cells = load_data(Path(config["data_folder"]), config["integrity_file"])
    all_rows, replays = [], []
    for result in results:
        folder = run / result["name"]
        bundle = joblib.load(folder / "selected_model.joblib")
        split = json.loads((folder / "split.json").read_text())
        if set(bundle["train_ids"]) != set(split["train"] + split["validation"]) or set(bundle["train_ids"]) & set(split["test"]):
            raise ValueError("saved model train/test provenance mismatch")
        report = json.loads((folder / "test/metrics.json").read_text())
        with np.load(folder / "test/predictions.npz") as saved:
            current = [r for r in report["by_cell"] if r["origin"] == 100]
            columns = min(4, max(1, len(current)))
            n_rows = max(1, int(np.ceil(len(current) / columns)))
            fig, axes = plt.subplots(n_rows, columns, figsize=(columns * 4.8, n_rows * 3), squeeze=False)
            for ax in axes.flat:
                ax.set_visible(False)
            for ax, row in zip(axes.flat, current):
                ax.set_visible(True)
                key = row["cell_id"]
                future = saved[f"{key}__100__cycle"]
                ax.plot(future, saved[f"{key}__100__truth"], color="black", lw=1.4, label="Observed future SOH")
                rejected = ~saved[f"{key}__100__regular_target"]
                if rejected.any():
                    ax.scatter(future[rejected], saved[f"{key}__100__truth"][rejected], color="#bb3344", s=18, marker="x", label="Interrupted diagnostic")
                ax.plot(future, saved[f'{key}__100__{bundle["selected_method"]}'], color="#16836a", lw=1.4, label="Selected forecast")
                ax.plot(future, saved[f"{key}__100__type_mean"], color="#c3822b", ls="--", lw=1, label="Type mean baseline")
                ax.set_title(f'{key} | MAPE {row["metrics"][bundle["selected_method"]]["mape_percent"]:.2f}%', fontsize=10)
                ax.set_xlabel("Source cycle (forecast issued at 100)")
                ax.set_ylabel("SOH")
                ax.grid(alpha=.2)
            if current:
                axes.flat[0].legend(fontsize=7)
            fig.suptitle(result["name"] + " | whole-cell holdout; observed future only", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, .96))
            fig.savefig(folder / "future_soh_curves.png", dpi=140)
            plt.close(fig)
            for row in report["by_cell"]:
                key, origin = row["cell_id"], row["origin"]
                c = cells[key]
                take = c["cycle"] <= origin
                future = saved[f"{key}__{origin}__cycle"]
                predicted, _ = predict_curve(bundle, c["cycle"][take], c["raw"][take],
                    c["metadata"], origin, future, bundle["selected_method"])
                expected = saved[f'{key}__{origin}__{bundle["selected_method"]}']
                difference = float(np.max(np.abs(predicted - expected)))
                if difference > 1e-10:
                    raise ValueError("prefix-only saved-model replay failed")
                replays.append({"run": result["name"], "cell_id": key, "origin": origin,
                                "max_absolute_prediction_difference": difference})
                all_rows.append({**row, "run": result["name"], "dataset": result["dataset"],
                    "selected_method": bundle["selected_method"],
                    "selected_mape_percent": row["metrics"][bundle["selected_method"]]["mape_percent"]})
            if result["name"].endswith("repeat1") or result["name"] == "NASA_fold1":
                example = current[0]
                cell = cells[example["cell_id"]]
                keep = cell["cycle"] <= 100
                np.savez_compressed(folder / "example_signals.npz", cycle=cell["cycle"][keep], raw_features=cell["raw"][keep])
                save_json(folder / "example_metadata.json", cell["metadata"])
                save_json(folder / "example_provenance.json", {"test_cell": example["cell_id"], "origin": 100,
                    "labels_in_input": False, "requested_example_end_cycle": 200})
    summary = {domain: {str(origin): aggregate([r for r in all_rows if r["dataset"] == domain and r["origin"] == origin])
                       for origin in ORIGINS} for domain in DOMAINS}
    by_type = {key: aggregate([r for r in all_rows if r["type"] == key and r["origin"] == 100])
               for key in sorted({r["type"] for r in all_rows})}
    primary = [r for r in all_rows if r["origin"] == 100]
    summary_report = {"scope": "three fixed development holdouts for MIT/XJTU, four leave-one-cell-out NASA folds; repeated cells averaged first",
        "domains": summary, "types_at_origin100": by_type,
        "exact_protocol_coverage": {domain: {
            "seen": aggregate([r for r in primary if r["dataset"] == domain and r["exact_protocol_seen"]]),
            "unseen_within_seen_family": aggregate([r for r in primary if r["dataset"] == domain and not r["exact_protocol_seen"]])}
            for domain in DOMAINS},
        "by_cell_origin_run": all_rows}
    save_json(run / "summary.json", summary_report)
    save_json(run / "reload_verification.json", {"verified_origins": len(replays), "checks": replays,
        "input_fields": "prefix cycle/raw_features + known metadata + requested future cycle", "future_signals_used": False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.8]})
    for domain, color in zip(DOMAINS, ("#1f77b4", "#16836a", "#c3822b")):
        axes[0].plot(ORIGINS, [summary[domain][str(o)]["cell_macro_mape_percent"] for o in ORIGINS], marker="o", label=domain, color=color)
    axes[0].axhline(6, color="#bb3344", ls="--", label="6% target")
    axes[0].set(xlabel="Forecast origin (source cycle)", ylabel="Full observed-future MAPE (%)", title="Whole-cell held-out forecasts")
    axes[0].legend()
    labels = list(by_type)
    values = [by_type[k]["cell_macro_mape_percent"] for k in labels]
    axes[1].barh(labels, values, color=["#16836a" if v <= 6 else "#c3822b" for v in values])
    axes[1].axvline(6, color="#bb3344", ls="--")
    axes[1].set(xlabel="Cell-macro MAPE (%)", title="Known type: forecast from cycle 100")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(run / "overview.png", dpi=150)
    plt.close(fig)
    lines = ["# 同类型未见电池：未来 SOH 预测", "", "使用已检查过的公开数据做开发评测；不是新电池盲测。",
        "模型只使用预测起点前的信号和已知工况，未来容量仅用于训练电池标签或测试评分。",
        "以第 100 个源循环为主要预测起点，未来范围为记录里实际可观察的全部后续 SOH。", "",
        "| 数据集 | 不重复测试电池数 | 20 循环出发 MAPE | 50 循环出发 MAPE | 100 循环出发 MAPE | 100 循环出发 ≤6% 电池比例 |",
        "|---|---:|---:|---:|---:|---:|"]
    for domain in DOMAINS:
        s = summary[domain]
        lines.append(f'| {domain} | {s["100"]["unique_test_cells"]} | {s["20"]["cell_macro_mape_percent"]:.3f}% | {s["50"]["cell_macro_mape_percent"]:.3f}% | {s["100"]["cell_macro_mape_percent"]:.3f}% | {100 * s["100"]["fraction_cells_mape_le_6"]:.1f}% |')
    lines += ["", "## 预测跨度与末段（源循环 100 出发）", "",
        "各跨度只统计该范围内真实有诊断容量记录的电池；末段指每块电池可观察未来记录的后半段。",
        "", "| 数据集 | +10 循环 MAPE (电池数) | +50 循环 MAPE (电池数) | +100 循环 MAPE (电池数) | 末段 MAPE |",
        "|---|---:|---:|---:|---:|"]
    for domain in DOMAINS:
        s = summary[domain]["100"]
        horizons = []
        for h in (10, 50, 100):
            item = s["within_horizon"][str(h)]
            horizons.append(f'{item["cell_macro_mape_percent"]:.3f}% ({item["cells_with_observations"]})' if item["cell_macro_mape_percent"] is not None else "— (0)")
        lines.append(f'| {domain} | {horizons[0]} | {horizons[1]} | {horizons[2]} | {s["late_half"]["cell_macro_mape_percent"]:.3f}% |')
    lines += ["", "## 类型结果（源循环 100 出发）", "", "| 类型 | 电池数 | MAPE | ≤6% 比例 |", "|---|---:|---:|---:|"]
    for key, value in by_type.items():
        lines.append(f'| {key} | {value["unique_test_cells"]} | {value["cell_macro_mape_percent"]:.3f}% | {100 * value["fraction_cells_mape_le_6"]:.1f}% |')
    lines += ["", "## 使用范围", "", "- MIT 类型包括实验批次和恒定/递减/递增快充工况族；精确策略覆盖在 summary.json 中逐电池标明。",
        "- XJTU 按六个已知实验策略分类。诊断间隔大，20/50 循环可能历史不足，跳过原因逐项保存。",
        "- NASA 只有四块电池，同系列但放电截止电压不完全相同，只能作小样本补充。",
        "- MAPE 先按电池求均值；重复划分中重复出现的同一电池先平均，避免重复计权。",
        "- MIT 按原始时间/电流规则排除中断诊断记录（放电过程首尾跨度 >1 小时、相邻高倍率采样相隔 >60 秒等）；信号和标签清单见 target_exclusions.json，电池其余记录保留。",
        "- 原始未过滤未来容量仍随预测保存在 NPZ 中；各折 metrics.json 同时报 unfiltered_mape_percent。",
        "- 数据量少的类型、真实早衰和难例均保留；低平均 MAPE 不代表每块电池均达标。",
        "- 未来工况假设保持相应实验策略；不提供未经验证的独立 RUL。",
        "- 类型平均曲线在部分折可能由验证集选中；它不利用新电池信号调整曲线，必须如实视作基线。",
        "- 训练末端之外的预测缺乏同类标签支持，逐点支持数量随结果保存。", "",
        "## 结果与复现", "", "每个折目录保存训练/验证/测试 IDs、两组候选权重和验证结果、最终选定权重、测试预测及逐电池曲线。",
        "分类见 classification.json；质量清单位于 config.json 指向的数据目录 quality_audit.json。",
        f"全部 {len(replays)} 个测试预测起点已从保存的权重、仅含历史信号的输入重放核对。", "",
        "```powershell", r".\.venv\Scripts\python.exe -m scripts.prepare_same_type_forecast",
        r".\.venv\Scripts\python.exe -m scripts.audit_mit_discharge_integrity",
        r".\.venv\Scripts\python.exe -m scripts.train_same_type_forecast",
        r".\.venv\Scripts\python.exe -m scripts.report_same_type_forecast", "```", "",
        "原始 MAT 转信号前缀入口 scripts.extract_forecast_prefix：--source、--metadata、--origin、--output；MIT 另需 --cell-index，NASA 另需 --variable。",
        "新电池推理入口 scripts.predict_same_type_forecast：--model、--signals、--metadata、--origin、--end-cycle、--output。",
        "各首轮目录提供 example_signals.npz 和 example_metadata.json；输入没有容量、SOH 或寿命标签。"]
    (run / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({d: summary[d]["100"] for d in DOMAINS}, indent=2))
    print("SAVED", run / "README.md")


if __name__ == "__main__":
    main()
