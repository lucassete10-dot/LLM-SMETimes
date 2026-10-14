# 同类型新电池未来 SOH 预测：使用与评测

目标为输入同类型新电池截至固定循环的电压、电流、温度历史及已知工况，输出未来循环的 SOH。主要评测起点是源循环 100，另有 20、50 循环对照。未来工况假定维持该实验策略。容量、真实当前 SOH、最终寿命和未来波形均不属于推理输入。

## 已完成的开发评测

MIT/XJTU 各做三组固定整电池划分，每组每类约 20% 电池留作测试，剩余电池先划出验证集选模型，再合并训练和验证电池重训。NASA 做四组留一电池评测。测试电池全程不参与相应折训练和参数选择。公共数据此前已被查看，因此这些结果属于开发评测。

| 数据集/方法 | 不重复测试电池数 | 循环 100 起全部可观察未来的 cell-macro MAPE | 解释 |
|---|---:|---:|---|
| MIT，验证集选模型 | 72 | 1.571% | 三次划分，同一电池重复出现先平均 |
| XJTU，验证集选模型 | 33 | 1.908% | 同上，稀疏诊断按源循环计时 |
| NASA，原模型 | 4 | 8.558% | 两块明显偏移 |
| NASA，电压候选与原模型逐折验证选择 | 4 | 5.995% | B0007 仍为 14.235%，仅一块验证电池，选择不稳定 |
| NASA，固定电压候选 | 4 | 3.704% | 看过原误差后提出的探索性对照，尚无新电池独立检验 |

MIT、XJTU 各测试电池重复划分后的平均 MAPE 均小于 6%；最难电池分别为 b3c38（4.834%）和 b2_3C_battery-11（5.708%）。这不保证每次划分、每个未来点或之后的每块新电池均低于 6%。

MIT 分类是批次 × 恒定/递增/递减快充工况族；XJTU 分类是六个实验策略。NASA 只有四块同系列电池且截止电压不同，不能把它视为充分覆盖全部精确工况的同类大样本。类型计数和精确协议覆盖随评测报告保存。

## 结果入口

- 主结果：`battery_models_v4/same_type_forecast/runs/20260930T064130_235459Z/README.md`。
- 主汇总图：同目录 `overview.png`；每折 `future_soh_curves.png` 包含该折全部循环 100 测试电池曲线。
- 每块电池、每个起点的预测及真实曲线：各折 `test/predictions.npz`；分类型/数据集、短程和末段误差：`summary.json`；难例：`hard_cases.json`。
- NASA 对照：`battery_models_v4/nasa_voltage_forecast/runs/20260929T111233_566119Z/README.md`，另有 `overview.png` 和逐电池 `comparison.png`。
- 数据：`data/interim/same_type_forecast/20260929T120631_212616Z/manifest.json` 与 `quality_audit.json`。
- 浮点时间修正：`omitted_positive_mit_trace_audit_20260929T120629_371926Z.json`（在同一 interim 目录）记录27条误删标签的恢复、4条容量计数异常和1条不完整放电。恢复后的全部10折重新训练；旧实验保留。
- XJTU旧容量上下限筛选与原始参考容量复核：同一 interim 目录 `inherited_target_audit_20260929T120652_211575Z.json`，未发现因该上下限丢失的正容量可比诊断。
- 原始 MIT 放电完整性审计：`data/interim/same_type_forecast/mit_discharge_integrity_20260929T060206_688121Z.json`。
- 排除记录与输入历史：主结果目录 `target_exclusions.json`。24 个中断诊断容量记录不用于正常容量监督，6 个对应早期输入记录也已排除；电池本身保留。包含这些容量的未过滤 MAPE 同时保存。
- 复核：主结果目录 `delivery_verification.json`、`reload_verification.json`、`raw_input_verification.json`、`cli_verification.json`。

## 运行现有模型

在项目目录下运行 PowerShell。下面例子使用留出的 MIT 电池前缀，预测循环 101–200；输出文件必须尚不存在，防止覆盖历史结果。

```powershell
$forecastRun = 'battery_models_v4/same_type_forecast/runs/20260930T064130_235459Z'
.\.venv\Scripts\python.exe -m scripts.predict_same_type_forecast --model "$forecastRun/MIT_repeat1/selected_model.joblib" --signals "$forecastRun/MIT_repeat1/example_signals.npz" --metadata "$forecastRun/MIT_repeat1/example_metadata.json" --origin 100 --end-cycle 200 --output forecast_new_result.json
```

XJTU 将 `MIT_repeat1` 改成 `XJTU_repeat1`。NASA 原模型示例位于 `NASA_fold1`。NASA 电压候选统一使用同一推理命令，`--model` 可指向 NASA 对照目录下某折 `refit_model.joblib`；四块全部训练的试用权重是 `all_four_cells_exploratory_model.joblib`，后者不用于重复评价这四块训练电池。

MIT/XJTU 可直接以首轮保存的 `selected_model.joblib` 作为后续新电池试用模型；选择首轮是固定交付约定，不根据测试误差挑折。

输出 JSON 包含未来源循环、SOH 比值（1.0 表示 100%）、逐点同类型训练曲线支持数、未来工况假设及超出训练记录范围的标记。训练支持为零处仍可能给出数值，但该区间没有已验证的同类记录支持。

## 输入新的原始电池

先准备已知规格和协议的 metadata JSON，可参考各数据集例子。`dataset` 与 `group` 必须来自已知电池/实验信息，MIT 需要 `charge_protocol`，NASA 需要 `discharge_cutoff_v`。不可用新电池最终循环数指定类型。

原始 MAT 提取入口：

```powershell
.\.venv\Scripts\python.exe -m scripts.extract_forecast_prefix --source PATH_TO_NEW_CELL.mat --metadata new_metadata.json --origin 100 --output new_prefix.npz
```

MIT 批次文件追加 `--cell-index`（从 0 开始）；NASA 追加 `--variable B0018` 等真实 MATLAB 变量名。XJTU 使用单电池 MAT。提取结果只含 `cycle` 与 `raw_features`，不含容量、SOH 或寿命。再将该 NPZ 和 metadata 交给推理命令。

当前适配器对应这些公开数据的 MAT 结构；新的仪器格式需要先按相同信号约定适配。MIT 使用放电前 300 秒，XJTU 使用可比诊断充放电各前 120 秒，NASA 使用放电前 1000 秒。XJTU 稀疏诊断至少需要 6 次已观测诊断；不足时返回错误，不借用未来数据补齐。

## 重训和验证

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_same_type_forecast
.\.venv\Scripts\python.exe -m scripts.audit_mit_discharge_integrity
.\.venv\Scripts\python.exe -m scripts.train_same_type_forecast
.\.venv\Scripts\python.exe -m scripts.report_same_type_forecast
.\.venv\Scripts\python.exe -m unittest tests.test_same_type_forecast -v
```

准备和训练均新建时间戳目录。候选与选定模型完成后立即保存权重、预测和指标，评测函数同时保存曲线图；失败时记录异常。报告命令在该次目录中生成汇总报告。历史失败、被后续修正替代的实验也保留，其状态不能与成功主结果混用。

NASA 电压对照复现命令为 `python -m scripts.train_nasa_voltage_forecast`；该脚本明确绑定上述冻结主结果划分，以便保持对照可比性。随后对新目录运行 `python -m scripts.report_nasa_voltage_forecast --run NEW_RUN_DIRECTORY`。

当前功能不输出独立 RUL。SOH 误差验证只覆盖记录中真实可观察的未来范围；MIT、XJTU、NASA 的记录长度与诊断密度不同，不能把三个平均数解释成相同预测难度。
