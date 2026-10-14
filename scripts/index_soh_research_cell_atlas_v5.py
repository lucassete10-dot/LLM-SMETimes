"""Read-only per-battery atlas index and QA ledger; never draw or fit."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pymupdf
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.prepare_same_type_forecast import save_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve() / "results/soh_research_v5_cell_atlas"
    atlas = args.atlas.resolve()
    if not atlas.is_relative_to(root) or atlas == root:
        raise ValueError("explicit local atlas directory required")
    state, config = read(atlas / "progress.json"), read(atlas / "config.json")
    if not state["complete"] or len(state["completed"]) != state["expected_cells"] or (atlas / "failure.json").exists():
        raise ValueError("index cannot certify a partial or failed rendering batch")
    if stream_sha(atlas / "plot_source.py") != config["source_sha256"]:
        raise ValueError("atlas preserved drawing source changed")
    output = atlas / "indexes" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "index_source.py")
    lines = ["# 逐电芯未来SOH诊断图册", "", "全部冻结测试电芯，固定展示Ridge / ExtraTrees / GRU。不是按测试误差挑出的代表样本，也不是LLM最终结果。", "",
        "每张图从左到右使用前20/50/100源循环；阴影为训练种子SD，不是置信区间。各面板的未来目标区间不同，不能直接用来证明输入越多越好。XJTU温度缺失，NASA物理规格未核验且仅四芯补充。", "",
        "所有图已通过自动字号、对齐和碰撞检查；逐面板人工复核尚未覆盖全部。请把未复核页面视为诊断草稿，而非投稿就绪图。", "",
        "| 数据集 | 电芯 | N100 MAPE %：Ridge / ExtraTrees / GRU | 图 | 向量PDF |", "|---|---|---|---|---|"]
    records, members = [], set()
    for item in state["completed"]:
        key = (item["dataset"], item["cell_id"])
        if key in members:
            raise ValueError("duplicate atlas battery")
        members.add(key)
        folder = Path(item["folder"]).resolve()
        if folder != atlas / item["dataset"] / item["cell_id"]:
            raise ValueError("atlas cell path differs from explicit identity")
        alignment, text, collision = [read(folder / name) for name in ("alignment.json", "text-audit.json", "collision-audit.json")]
        if (alignment["verdict"] != "PASS" or not alignment["auditable"] or collision["verdict"] != "PASS" or not collision["auditable"]
                or not text["auditable"] or text["below_minimum_count"] or text["minimum_found_pt"] < 5):
            raise ValueError("cell figure has unresolved automated geometry QA")
        with pymupdf.open(folder / "curves.pdf") as document:
            if len(document) != 1:
                raise ValueError("per-cell export must contain one figure")
            dimensions = [document[0].rect.width*25.4/72, document[0].rect.height*25.4/72]
        if any(abs(actual-expected) > .01 for actual, expected in zip(dimensions, config["contract"]["dimensions_mm"])):
            raise ValueError("rendered PDF physical dimensions differ from figure contract")
        numbers = read(folder / "numbers.json")
        metric = {row["model"]: row["mape_mean"] for row in numbers["numbers"] if row["origin"] == 100}
        score = " / ".join(f"{metric[model]:.2f}" if metric[model] is not None else "不适用" for model in config["models"])
        lines.append(f'| {item["dataset"]} | {item["cell_id"]} | {score} | [查看]({(folder / "curves.png").as_posix()}) | [PDF]({(folder / "curves.pdf").as_posix()}) |')
        records.append({"dataset": item["dataset"], "cell_id": item["cell_id"], "dimensions_mm": dimensions,
            "minimum_glyph_pt": text["minimum_found_pt"], "automated_geometry_qa": "passed",
            "visual_review_completed": False, "mape_n100": metric,
            "artifact_sha256": {name: stream_sha(folder / name) for name in ("curves.png", "curves.pdf", "curves.svg", "source_data.npz", "numbers.json", "alignment.json", "text-audit.json", "collision-audit.json")}})
    save_json(output / "qa_ledger.json", {"atlas": str(atlas), "atlas_config_sha256": stream_sha(atlas / "config.json"),
        "atlas_progress_sha256": stream_sha(atlas / "progress.json"), "index_source_sha256": stream_sha(__file__), "records": records,
        "automated_geometry_qa_completed_cells": len(records), "visual_reviews": "separate explicit manual review ledger; not inferred from automated pass",
        "source_preflight_warnings": "PNG400dpi is a diagnostic preview, not a submission TIFF; source parser misreads arithmetic figsize as inches, actual PDF dimensions measured here"})
    (output / "index.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"folder": str(output), "batteries": len(records), "all_pdf_dimensions_verified": True, "visual_review_required": True}, indent=2))


if __name__ == "__main__":
    main()
