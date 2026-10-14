"""Freeze future condition experiments without fitting or touching live runs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from battery_smetimes.physical_types import load_registry
from battery_smetimes.research_generalization_splits import make_condition_splits, metadata_only
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-folder", type=Path)
    args = parser.parse_args()
    audit = args.audit_folder or Path(json.loads(Path(
        "experiments/soh_research_v5/latest_audit.json").read_text(encoding="utf-8"))["folder"])
    frozen = json.loads((audit / "config.json").read_text(encoding="utf-8"))
    registry_path = audit / "source_snapshot/battery_physical_registry.json"
    if sha(registry_path) != frozen["registry_sha256"]:
        raise ValueError("frozen physical registry changed")
    records_path = audit / "cell_registry.json"
    records = metadata_only(json.loads(records_path.read_text(encoding="utf-8"))["cells"])
    registry = load_registry(registry_path)
    plans = make_condition_splits(records, frozen["config"], registry)
    main_runs = []
    for run in sorted(Path("experiments/soh_research_v5/runs").glob("*")):
        source = run / "config.json"
        if not source.is_file() or not (run / "progress.json").is_file():
            continue
        config = json.loads(source.read_text(encoding="utf-8"))
        if config["smoke"]:
            continue
        if sha(run / "frozen_audit/config.json") != sha(audit / "config.json"):
            raise ValueError("main run refers to a different frozen audit")
        main_runs.append({"run": str(run.resolve()), "config_sha256": sha(source),
                          "source_sha256": config["source_sha256"]})
    if not main_runs:
        raise ValueError("no frozen main-model runs to bind condition experiment budgets")
    folder = Path("experiments/soh_research_v5/condition_plans") / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    sources = [Path("scripts/prepare_soh_research_conditions_v5.py"),
               Path("battery_smetimes/research_generalization_splits.py"),
               Path("battery_smetimes/research_splits.py"),
               Path("battery_smetimes/physical_types.py"), Path("battery_smetimes/forecast.py")]
    hashes = {}
    for source in sources:
        target = folder / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source)] = sha(source)
    save_json(folder / "config.json", {"status": "frozen metadata-only plans; no training or results",
        "frozen_audit": str(audit.resolve()), "audit_config_sha256": sha(audit / "config.json"),
        "source_cell_registry_sha256": sha(records_path), "registry_sha256": sha(registry_path),
        "prepared_manifest_sha256": frozen["prepared_manifest_sha256"],
        "integrity_sha256": frozen["integrity_sha256"], "source_sha256": hashes,
        "origins_source_cycles": frozen["config"]["origins_source_cycles"],
        "training_seeds": frozen["config"]["training_seeds"],
        "split_seed": frozen["config"]["split_seed"],
        "validation_fraction_of_development": frozen["config"]["validation_fraction_of_development"],
        "main_runs_with_frozen_model_search_and_source": main_runs,
        "execution_gate": "Fit only after the same-type main comparison/ablations are complete; smoke before full run.",
        "nasa_exclusion_from_this_supplement": "four unverified cells and cutoff groups of one/two do not support a reliable additional leave-condition-out study; existing LOBO retained",
        "not_cross_dataset": "No target family is overwritten. Two-direction cross-dataset prediction still needs a separate justified shared-signal/transfer protocol.",
        "prior_data_exposure": "These public cells were previously inspected; this is not new blinded validation."})
    save_json(folder / "metadata_only.json", {"cells": records})
    save_json(folder / "splits.json", {"plans": plans})
    lines = ["# 预设跨工况整电芯划分（尚未训练）", "",
        "仅使用已知实验工况，不读容量、SOH、寿命、信号或测试误差。物理家族不改变。MIT留出的是两阶段充电倍率的增/减/恒定模式，不等同于具体数值倍率全部未见；XJTU按作者描述的使用工况留出。NASA继续保留原四芯LOBO补充。", "",
        "| 划分 | 完整留出工况 | 训练芯 | 验证芯 | 测试芯 |", "|---|---|---:|---:|---:|"]
    for plan in plans:
        lines.append(f'| {plan["name"]} | {plan["heldout_condition"]} | {len(plan["train"])} | {len(plan["validation"])} | {len(plan["test"])} |')
    lines.extend(["", "模型、种子与调参预算延用主实验预设；验证只来自非留出工况。当前只有划分，不是跨工况精度结果；待主对比与消融完成后执行。", ""])
    (folder / "condition_plan.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(Path("experiments/soh_research_v5/latest_condition_plan.json"), {"folder": str(folder.resolve())})
    print(json.dumps({"folder": str(folder.resolve()), "plans": len(plans),
        "status": "metadata-only plans frozen; no model fitted"}, indent=2))


if __name__ == "__main__":
    main()
