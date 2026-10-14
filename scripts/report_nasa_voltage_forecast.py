"""Summarize the frozen NASA voltage challenger without hiding model-selection failures."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.prepare_same_type_forecast import save_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    folds = result["folds"]
    names = [row["test_cell"] for row in folds]
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, field, label, color in (
        (-.25, "prior_test_mape_percent", "Prior model", "#c28320"),
        (0, "candidate_test_mape_percent", "Voltage candidate (post-hoc)", "#1976a3"),
        (.25, "validation_selected_test_mape_percent", "Validation-selected", "#16836a")):
        ax.bar(x + offset, [row[field] for row in folds], width=.24, label=label, color=color)
    ax.axhline(6, color="#bd3845", ls="--", label="6% target")
    ax.set_xticks(x, names)
    ax.set(ylabel="Full observed-future MAPE (%)", xlabel="Held-out NASA cell",
           title="Cycle-100 NASA SOH forecast: 4-cell development comparison")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(run / "overview.png", dpi=160)
    plt.close(fig)
    sources = run / "source_snapshot"
    sources.mkdir(exist_ok=True)
    hashes = {}
    for name in ("scripts/train_nasa_voltage_forecast.py", "scripts/report_nasa_voltage_forecast.py",
                 "scripts/predict_same_type_forecast.py", "scripts/verify_same_type_cli.py"):
        shutil.copy2(name, sources / Path(name).name)
        hashes[name] = sha(name)
    save_json(run / "source_snapshot.json", hashes)
    lines = ["# NASA 电压特征对照实验", "",
        "此实验由前一版 NASA 测试曲线的系统性偏移启发；四块电池都已被查看，属于开发集对照，不是新的独立盲测。",
        "使用第 100 个源循环之前最近三次放电中 1000 秒处电压的中位数、预测起点和未来循环跨度；不输入新电池容量或未来信号。",
        "原划分不变：每折训练 2 块、验证 1 块、测试 1 块。仅用验证误差决定该折是采用旧模型还是电压候选，然后在训练+验证的 3 块上重训候选。", "",
        "| 测试电池 | 旧模型 MAPE | 电压候选 MAPE | 验证集选中的模型 | 选中模型 MAPE |",
        "|---|---:|---:|---|---:|"]
    for row in folds:
        lines.append(f'| {row["test_cell"]} | {row["prior_test_mape_percent"]:.3f}% | {row["candidate_test_mape_percent"]:.3f}% | {row["selected_by_validation"]} | {row["validation_selected_test_mape_percent"]:.3f}% |')
    lines += ["", f'旧模型四电池平均 **{result["prior_four_cell_macro_mape_percent"]:.3f}%**；固定电压候选四电池平均 **{result["candidate_four_cell_macro_mape_percent"]:.3f}%**；逐折验证选择平均 **{result["validation_selected_four_cell_macro_mape_percent"]:.3f}%**。',
        "", "B0007 的验证集偏向旧模型，因此按验证规则测试误差仍为 14.235%；不能看见测试结果后改选电压模型。",
        "固定电压候选的 3.704% 是探索性比较，不应作为对未见第五块 NASA 电池的承诺。不同放电截止条件、只有四块电池且各折只有一块验证电池，使模型选择不稳定。",
        "all_four_cells_exploratory_model.joblib 训练于现有四块电池，供未来新电池试用；其准确度尚待用户后续独立电池数据检验。", "",
        "每折保留验证/重训权重、预测数组、指标和单电池比较图；overview.png 展示上述三种口径。",
        "统一推理入口 scripts.predict_same_type_forecast 可读取电压模型及仅含 cycle/raw_features 的输入。"]
    (run / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SAVED", run / "README.md")


if __name__ == "__main__":
    main()
