"""Audit existing artifacts before creating a metadata-only frozen research plan."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.io import loadmat

from battery_smetimes.forecast import prefix_vector, type_key
from battery_smetimes.physical_types import (compare_specifications,
    family_for_registered_record, load_registry, match_external_specifications)
from battery_smetimes.research_splits import condition_stratum, make_research_splits
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data


def audit(config_file):
    config = json.loads(config_file.read_text(encoding="utf-8"))
    registry_file = Path(config["registry"])
    registry = load_registry(registry_file)
    data_folder = Path(json.loads(Path(config["prepared_data_pointer"]).read_text())["folder"])
    manifest, cells = load_data(data_folder, config["integrity_audit"])
    quality = json.loads((data_folder / "quality_audit.json").read_text(encoding="utf-8"))
    quality_by_id = {row["cell_id"]: row for row in quality["by_cell"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = Path("experiments/soh_research_v5/audits") / stamp
    folder.mkdir(parents=True, exist_ok=False)
    records, per_cell = {}, []
    for source in manifest["cells"]:
        key, meta = source["cell_id"], source["metadata"]
        family = family_for_registered_record(source, registry)
        qrow, cell = quality_by_id[key], cells[key]
        reference_cycle = max(qrow["reference_only_cycles"], default=0)
        if reference_cycle <= 0:
            raise ValueError(f"{key}: reference establishment cycle not recorded")
        # Reference is a label definition only, never an inference feature.
        with np.load(source["label_source"]) as labels:
            source_cycles = labels["cycle"].astype(int)
            expected_soh = labels["capacity_ah"].astype(float) / source["reference_capacity_ah"]
            shared, si, pi = np.intersect1d(source_cycles, cell["label_cycle"], return_indices=True)
            finite = np.isfinite(cell["soh"][pi])
            if not np.allclose(cell["soh"][pi][finite], expected_soh[si][finite], rtol=1e-5, atol=1e-5):
                raise ValueError(f"{key}: capacity/reference label mismatch")
        records[key] = {"cell_id": key, "metadata": meta, "physical_family": family,
            "physical_status": registry["families"][family]["status"],
            "condition_stratum": condition_stratum(meta), "prepared_file": source["file"],
            "prepared_sha256": source["sha256"], "raw_source": source["raw_source"],
            "reference_capacity_ah": source["reference_capacity_ah"],
            "reference_rule": source["reference_rule"], "reference_established_source_cycle": reference_cycle}
        eligibility = {}
        for origin in config["origins_source_cycles"]:
            reasons = []
            if reference_cycle > origin:
                reasons.append("label_reference_not_established_at_origin")
            try:
                prefix_vector(cell["cycle"], cell["raw"], meta, origin, [type_key(meta)])
            except ValueError as error:
                reasons.append(str(error))
            future = np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin)
            if future.sum() < 2:
                reasons.append("less_than_two_valid_future_targets")
            eligibility[str(origin)] = {"eligible": not reasons, "reasons": reasons,
                "observed_signal_records": int((cell["cycle"] <= origin).sum()),
                "valid_future_targets": int(future.sum())}
        per_cell.append({"cell_id": key, "dataset": meta["dataset"], "physical_family": family,
            "source_cycle_min": int(cell["label_cycle"].min()),
            "source_cycle_max": int(cell["label_cycle"].max()),
            "valid_target_count": int(np.isfinite(cell["soh"]).sum()),
            "signal_prefix_count": len(cell["cycle"]),
            "reference_established_source_cycle": reference_cycle,
            "restored_label_cycles": source["restored_source_label_cycles"],
            "excluded_interrupted_input_cycles": cell["excluded_input_cycles"],
            "eligibility": eligibility})
    # Never pass measured arrays, targets, lifetimes or learned statistics here.
    plans = make_research_splits(records, config, registry)
    summaries = {}
    for domain in ("MIT", "XJTU", "NASA"):
        rows = [row for row in per_cell if row["dataset"] == domain]
        sources = [source for source in manifest["cells"] if source["metadata"]["dataset"] == domain]
        summaries[domain] = {"cells": len(rows),
            "per_cell_source_last_cycle_range": [min(r["source_cycle_max"] for r in rows), max(r["source_cycle_max"] for r in rows)],
            "per_cell_valid_target_count_range": [min(r["valid_target_count"] for r in rows), max(r["valid_target_count"] for r in rows)],
            "reference_capacity_ah_range": [min(r["reference_capacity_ah"] for r in sources), max(r["reference_capacity_ah"] for r in sources)],
            "conditions": dict(Counter(condition_stratum(r["metadata"]) for r in sources)),
            "eligible_cells_by_origin": {str(n): sum(r["eligibility"][str(n)]["eligible"] for r in rows)
                                        for n in config["origins_source_cycles"]}}
    # NASA ambient conditions are observed in its first 100 discharge records only.
    nasa_ambient = {}
    for source in (r for r in manifest["cells"] if r["metadata"]["dataset"] == "NASA"):
        raw = loadmat(source["raw_source"], variable_names=[source["cell_id"]], simplify_cells=True)
        discharge = [r for r in raw[source["cell_id"]]["cycle"] if r["type"] == "discharge"][:100]
        nasa_ambient[source["cell_id"]] = sorted({float(r["ambient_temperature"]) for r in discharge})
    old_run = Path(json.loads(Path("battery_models_v4/same_type_forecast/latest.json").read_text())["run"])
    old_verified = json.loads((old_run / "delivery_verification.json").read_text())
    old_plans = json.loads((old_run / "splits.json").read_text())
    if isinstance(old_plans, dict):
        old_plans = old_plans.get("folds", old_plans.get("splits", []))
    old_overlap = []
    for plan in old_plans:
        partitions = [set(plan[p]) for p in ("train", "validation", "test")]
        if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
            old_overlap.append(plan["name"])
    families = list(registry["families"])
    pair_checks = [{"left": a, "right": b, **compare_specifications(
        registry["families"][a]["specifications"], registry["families"][b]["specifications"])}
        for i, a in enumerate(families) for b in families[i + 1:]]
    if any(r["status"] != "incompatible" for r in pair_checks if "A123" in r["left"]):
        raise ValueError("MIT physical incompatibility gate not established")
    report = {"status": "initial_audit_complete_full_research_pending", "prepared_folder": str(data_folder),
        "prepared_manifest_sha256": sha(data_folder / "manifest.json"), "data_cells_verified": len(records),
        "summaries": summaries, "per_cell": per_cell, "physical_pair_checks": pair_checks,
        "nasa_prefix_ambient_temperature_c": nasa_ambient,
        "old_run": str(old_run), "old_delivery_checks_passed": old_verified["passed"],
        "old_partition_overlap_folds": old_overlap,
        "exclusions": quality["known_cell_exclusions"],
        "restored_positive_target_count": sum(len(row["restored_label_cycles"]) for row in per_cell),
        "frozen_fold_count": len(plans), "training_seeds": config["training_seeds"],
        "normalization": "SOH 0-1 relative initial comparable capacity; source-specific reference protocol kept explicit. No global fitted signal scaler. Existing imputer fits listed training cells, selected models may refit train+validation without test.",
        "type_audit": "Old dataset/batch/condition labels were not genuine intrinsic specifications. New registry uses author/official specifications, no learned clustering or test trajectories. XJTU part number and NASA chemistry/model remain unknown.",
        "llm_audit": "Original pretrained SMETimes is distinct from numeric derivatives/tree. Llama/GPT2/OPT experts are dense; softmax L1 is constant and multiplied by zero. Genuine SOH regression/matched ablations and five seeds remain pending.",
        "leakage_conclusion": "No whole-cell overlap in the authoritative old forecast folds. Source/CLI replay checks archived. This is not certification of every older v2/v3 exploratory experiment or test-selection independence; previously inspected public data stay development data.",
        "external_gate": {key: match_external_specifications(row["specifications"], registry)
                          for key, row in registry["families"].items()}}
    if old_overlap or not old_verified["passed"]:
        raise ValueError("authoritative old run failed leakage/integrity audit")
    save_json(folder / "audit.json", report)
    save_json(folder / "cell_registry.json", {"cells": records})
    save_json(folder / "splits.json", {"config": config, "plans": plans})
    save_json(folder / "config.json", {"config": config, "config_sha256": sha(config_file),
        "registry_sha256": sha(registry_file), "integrity_sha256": sha(config["integrity_audit"]),
        "prepared_manifest_sha256": sha(data_folder / "manifest.json")})
    snapshot = folder / "source_snapshot"
    for file in (config_file, registry_file, Path(__file__), Path("battery_smetimes/physical_types.py"),
                 Path("battery_smetimes/research_splits.py"), Path("tests/test_physical_types.py")):
        destination = snapshot / file.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
    lines = ["# 电池未来 SOH：修订研究初始审计（2026-09-30）", "",
        "状态：初始审计与物理注册/划分已落盘；不是五折五种子完成报告。所有公共电池此前已查看，只能称开发评测。", "",
        "## 数据规模与实际规格", "",
        "| 来源 | 电芯数 | 每芯末源循环范围 | 可用目标记录范围 | 物理规格与温度 |",
        "|---|---:|---|---|---|" ]
    descriptions = {
        "MIT": "A123 APR18650M1A，LFP/graphite，18650，1.1 Ah / 3.3 V，2.0–3.6 V，30°C",
        "XJTU": "LISHEN NCM532，18650，2 Ah / 3.6 V，2.5–4.2 V，室温（不编造25°C），型号编号未知",
        "NASA": "18650 / 2 Ah，化学体系/厂家/型号/额定电压未知；前100放电记录的环境温度逐芯见audit.json；独立未核验组"}
    for domain, summary in summaries.items():
        lines.append(f'| {domain} | {summary["cells"]} | {summary["per_cell_source_last_cycle_range"]} | {summary["per_cell_valid_target_count_range"]} | {descriptions[domain]} |')
    lines += ["", "MIT 134颗是本地审计后的源电芯规模，不等同于原论文124颗分析子集；不因寿命短/预测难删除其余电芯。XJTU实际有55颗，NASA仅4颗。诊断记录数量不等于源循环数。", "",
        "资料：[MIT作者稿](https://escholarship.org/uc/item/9532z8t0)、[同作者后续实验Methods](https://web.mit.edu/braatzgroup/Attia_Nature_2020.pdf)、[XJTU作者发布](https://zenodo.org/records/10963339)、[NASA官方](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)。字段来源和未核验边界在 configs/battery_physical_registry.json。", "",
        "## 工况、容量参考与早期可评估性", "",
        "MIT逐芯快充策略与4C诊断放电；XJTU六种已知策略，1C或0.5C可比诊断；NASA深放电截止电压是工况，不是化学体系。工况和批次分层不构成新的物理型号。", "",
        "SOH=可比诊断容量/初始可比参考容量，统一0–1；不是统一的容量/额定容量。MIT/NASA取首次可比完整诊断，XJTU取前五次可比诊断中位数，参考值和建立源循环逐芯保存。不把参考容量、当前/未来容量、SOH或EOL作为模型输入。早于参考建立或信号不足的起点不评分，明确披露覆盖，不移动起点凑分。", "",
        "| 来源 | N=20可评电芯 | N=50 | N=100 |", "|---|---:|---:|---:|"]
    for domain, summary in summaries.items():
        coverage = summary["eligible_cells_by_origin"]
        lines.append(f'| {domain} | {coverage["20"]} | {coverage["50"]} | {coverage["100"]} |')
    lines += ["", "## 完整性与泄漏", "",
        f'本轮逐一校验193份准备文件指纹及原标签容量/参考关系。恢复的MIT微小浮点时间逆序误删标签共{report["restored_positive_target_count"]}条。源清单不含5颗Batch2延续重复电芯和作者记录错误b3c37；时间/电流证据排除24个中断放电标签，不按模型误差排除电芯。XJTU非可比工况记录是使用记录，不是坏数据。', "",
        "权威旧未来SOH实验10个划分的train/validation/test电芯没有交集；以前的准备/369个预测前缀重放/原始适配/CLI检查保留。旧中位数插补器仅拟合训练电芯，验证选模型后允许train+validation重拟合且不读test。本初始审计不替所有历史v2/v3探索结果背书，也不把已看过的公开测试数据称作独立盲测。", "",
        "新类型注册依赖非学习的物理规格与来源证据，不用测试曲线聚类、测试容量或最终寿命。MIT与NASA额定容量等明确冲突，门禁拒绝合并。NASA和XJTU虽然都是2Ah/18650，也不能因两项相同证明同一型号。外部新电池缺失化学体系/型号等字段返回needs_specs，不接受仅填写数据集名。", "",
        f'新的MIT/XJTU五折与NASA留一共{len(plans)}个外折已冻结，训练种子{config["training_seeds"]}。每颗只在一个外折测试，内层验证按整电芯留出；小工况的精确协议可能仍未见，后续单列支持覆盖。种子重复的是训练，不是每个模型另挑有利划分。', "",
        "## 模型审计与下一步", "",
        "现有ExtraTrees与数值RegularizedSMETimes不是预训练LLM。原SMETimes默认专家数1；当前专家为密集计算。门控softmax的L1常量正则又乘0，不产生有效正则。新回归头、正确专家消融及参数/冻结控制待实现，旧树模型低误差不能证明LLM有贡献。", "",
        "下一步单折单种子基线冒烟，然后按冻结计划跑公平基线/小序列模型/真正LLM及消融。测试评分不得反馈选超参数；未赢如实报告。五折五种子完成前不公布其均值±标准差。", "",
        f'完整逐电芯审计/注册/冻结计划：{folder.as_posix()}。']
    document = "\n".join(lines) + "\n"
    (folder / "audit_report.md").write_text(document, encoding="utf-8")
    # The first initial report is the root entry point; later audits stay timestamped.
    root_report = Path("audit_report.md")
    if not root_report.exists():
        root_report.write_text(document, encoding="utf-8")
    save_json(Path("experiments/soh_research_v5/latest_audit.json"), {"folder": str(folder.resolve())})
    print(json.dumps({"audit_folder": str(folder.resolve()), "cells": len(records),
        "folds": len(plans), "coverage": {k: v["eligible_cells_by_origin"] for k, v in summaries.items()}}, indent=2), flush=True)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/soh_research_v5.json"))
    audit(parser.parse_args().config)
