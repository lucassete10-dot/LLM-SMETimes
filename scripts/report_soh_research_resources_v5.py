"""Keep contended engineering timings separate from idle-checked measurements."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from scripts.prepare_same_type_forecast import save_json, sha


def collect(folder):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if not (folder / "report.json").is_file():
        failure = json.loads((folder / "failure.json").read_text(encoding="utf-8")) if (folder / "failure.json").is_file() else None
        return {"folder": str(folder.resolve()), "status": "failed" if failure else "in_progress",
                "config_sha256": sha(folder / "config.json"), "failure": failure}, []
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    model = Path(config["model_folder"])
    for name, expected in report["evidence"]["artifact_sha256"].items():
        if sha(model / name) != expected:
            raise ValueError("benchmark points to modified model artifacts")
    records = []
    for case in report["cases"]:
        records.append({"benchmark_folder": str(folder.resolve()), "report_sha256": sha(folder / "report.json"),
            "model": model.name, "model_folder": str(model.resolve()), "device": config["device"],
            "formal_resource_comparison_eligible": report["formal_resource_comparison_eligible"],
            "total_parameters": report["model"]["total_parameters"], "trainable_parameters": report["model"]["trainable_parameters"],
            "backbone_artifact_bytes": report["required_shared_backbone_artifact_bytes"], **case})
    return {"folder": str(folder.resolve()), "status": report["status"],
        "report_sha256": sha(folder / "report.json"), "config_sha256": sha(folder / "config.json")}, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, action="append")
    args = parser.parse_args()
    folders = args.benchmark or sorted(Path("experiments/soh_research_v5/resource_benchmarks").glob("*"))
    records, runs = [], []
    for folder in folders:
        if not (folder / "config.json").is_file():
            continue
        run, rows = collect(folder)
        runs.append(run)
        records.extend(rows)
    output = Path("results/soh_research_v5_resources") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "report_source.py")
    save_json(output / "resource_report.json", {"runs": runs,
        "idle_checked_cases": [row for row in records if row["formal_resource_comparison_eligible"]],
        "contended_engineering_cases_not_comparative_evidence": [row for row in records if not row["formal_resource_comparison_eligible"]],
        "report_source_sha256": sha(Path(__file__)),
        "limits": "Rows retain individual batteries and technical repeats. No latency ranking, significance, independent-cell sample inflation, or causal claim about speedup. Timings exclude raw signal extraction/acquisition and model loading; those boundaries are in the benchmark reports. A single idle-checked run is still not a full-fold/seed resource comparison."})
    lines = ["# 保存模型资源测量", "", "重复计时是同电芯上的技术重复，不是新增独立电芯。首次统计卡片与复用缓存分开；总参数与可训练参数分开。测量从已经提取的早期特征开始，不包含原始电压/电流采集与特征提取。", ""]
    for eligible, heading in ((True, "空闲条件检查通过的测量"), (False, "有竞争负载的工程验证（不作为速度排名）")):
        own = [row for row in records if row["formal_resource_comparison_eligible"] == eligible]
        lines.extend([f"## {heading}", ""])
        if not own:
            lines.extend(["暂无；不能把正在训练时的计时升级为正式资源对比。", ""])
            continue
        lines.extend(["| 模型 | 电芯 | 设备 | 未来查询点数 | 缓存状态 | 技术重复数 | 中位数 ms | 技术SD ms | 总/可训练参数 |", "|---|---|---|---:|---|---:|---:|---:|---|"])
        for row in own:
            params = "树/核结构，不换算神经参数" if row["total_parameters"] is None else f'{row["total_parameters"]:,}/{row["trainable_parameters"]:,}'
            deviation = "未定义" if row["sd_ms"] is None else f'{row["sd_ms"]:.3f}'
            lines.append(f'| {row["model"]} | {row["cell_id"]} | {row["device"]} | {row["future_query_points"]} | {row["cache_mode"]} | {row["technical_repeats"]} | {row["median_ms"]:.3f} | {deviation} | {params} |')
        lines.append("")
    lines.extend(["## 失败与边界", "", f'失败记录{sum(run["status"] == "failed" for run in runs)}次，未完成记录{sum(run["status"] == "in_progress" for run in runs)}次；全部详情保留在JSON。默认空闲门禁拒绝训练占用属于预期保护，不是模型精度失败。', "", "只有适用的LLM统计卡片分支实际使用该缓存；其他模型的冷/热标签不代表存在缓存。冻结骨干仍需加载/驻留；adapter文件小不等于总部署模型小。单点查询是未来SOH，不是实时SOH估计；固定100点网格也不是知道电芯真实总寿命后的完整曲线。", ""])
    (output / "resource_table.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(Path("results/latest_soh_research_v5_resources.json"), {"folder": str(output.resolve())})
    print(json.dumps({"folder": str(output.resolve()), "idle_checked_cases": sum(row["formal_resource_comparison_eligible"] for row in records),
        "contended_cases": sum(not row["formal_resource_comparison_eligible"] for row in records)}, indent=2))


if __name__ == "__main__":
    main()
