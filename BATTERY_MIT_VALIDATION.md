# MIT 电池 SOH 最小验证（2026-09-23）

数据来自 Severson 等人的 MIT/Stanford/Toyota 电池循环寿命项目。官方入口为 <https://data.matr.io/1/>，作者代码为 <https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation>。本实验实际下载的是 Kaggle 镜像中的原始 Batch 1 MAT（非第三方预提取特征），镜像页：<https://www.kaggle.com/datasets/rickandjoe/mit-battery-degradation-dataset>。原始 MAT 的 SHA-256 为 `9d928ab978f0e3c70b31cb833a749fedd35094d01af76475d69b40aa3497f5ba`。只使用 Batch 1，不能代表整个 MIT 数据集的跨批次泛化。

## 标签与输入

- SOH = 当前循环放电容量 / 该电池首个有效循环放电容量。它不是预测未来 RUL，也不是论文中按 0.88 Ah 截止定义的循环寿命。
- 输入严格限于当前循环主放电段前 300 秒的电压、电流、温度。完整循环的容量及放电结束时间仅供标签与质量核对，不进入特征。
- 共同特征 15 维（电压/电流 4 个时间点及 7 个统计量）；增强特征 51 维（更密的采样、温度、斜率、分位差和积分）。同一电池各循环相对首个有效循环的特征差用于建模。
- 原始 46 个电池保留 38,747 个循环；剔除 46 个无效容量、16 个无效前缀和 2 个明显过高的容量值。详细记录在 `D:\LLM\MIT\processed\audit.csv`。

## 验证协议与结果

第一步：只用 NASA + CALCE 电池训练 300 秒、15 维输入的同结构数值版 SMETimes，再直接预测 MIT 46 个电池。由于 MIT 的 4C 放电段通常不足原 NASA 版本使用的 1,000 秒，**这不是原先 1,000 秒权重的直接推理**，而是对齐输入后重新训练的跨数据集迁移测试。

第二步：MIT 按电池而不是循环随机划分为训练 28、验证 8、测试 10（种子 2026）。测试 10 个电池共 8,372 个评估循环，均跳过各电池头 15 个循环；测试电池没有进入缺失值填充、缩放、岭回归、训练或选 epoch。训练、验证和测试 IDs 在 `battery_mit_results/mit_cell_holdout/split.json`。神经模型只以 1 个验证电池选 epoch，另外 7 个未用于选模；这一设计限制了对稳定性的判断。

| 模型 | MIT 训练前 / 46 电池 MAE | MIT 训练后 / 10 个未见电池 MAE | 训练后 R²（逐电池平均） |
| --- | ---: | ---: | ---: |
| 300 秒数值版 SMETimes，15 维 | 0.09183 | 0.00865 | 0.918 |
| 15 维随机森林 | 0.14247 | 0.00748 | 0.923 |
| 51 维 ExtraTrees | — | **0.00714** | 0.900 |
| 51 维数值版 SMETimes + 平方误差 | — | 0.00870 | 0.903 |
| 51 维数值版 SMETimes + Huber / 异常降权 | — | 0.00867 | 0.903 |

MAE 是 SOH 的绝对误差；`0.00714` 即平均约 **0.714 个 SOH 百分点**。宏平均为先对每个测试电池算误差、再平均，避免长序列占主导。所有精确值、逐电池值和逐循环预测见 `battery_mit_results`。

增强特征的树模型比 15 维森林 MAE 略低（0.00748 → 0.00714），但其逐电池平均 R² 反而下降。尤其在测试集中 SOH < 0.90 的 690 个循环上，增强树模型 MAE 为 0.0122，基础随机森林为 0.0098；第 95 百分位的绝对误差也分别为 0.0242 和 0.0211。因此不能概括为全面优于基线。数值版 SMETimes 及鲁棒损失在自然 MIT 数据上的精度**未超过**树模型；自然训练数据的局部异常降权触发次数为 0。额外在 3% 训练标签上人工注入 ±0.15 SOH 错误的压力测试，干净验证电池上 MSE 与鲁棒模型 MAE 分别为 0.007198 与 0.007197，改善可忽略；这不能证明真实异常场景下有效。

## 运行

已下载数据在 `D:\LLM\MIT\raw`；处理数据在 `D:\LLM\MIT\processed`。

```powershell
.\.venv\Scripts\python.exe prepare_mit.py
.\.venv\Scripts\python.exe battery_mit_validation.py
.\.venv\Scripts\python.exe mit_noise_stress.py
```

局限：只测试 Batch 1 的同批未见电池，未做 Batch 2/3 或充电策略迁移；部分 Batch 1 电池很早停止，SOH 覆盖范围不等。SOH 使用完整循环容量作事后监督标签，模型输入虽只看前 300 秒，但这个实验不等于在线实时容量测量，也不等于 RUL 预测。
