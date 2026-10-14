# LLM + SMETimes v3 五数据集执行结果

五个数据集完成51/51正式槽位，固定三个种子。实际新拟合66次：5次快速R0、10次A/B筛选、51次正式配置（包含复用的15次R0确认）。所有数据集验证锁定配置均为R0。全部最终TRAIN/VAL/TEST图已审阅并核对原始预测。完整目标的逐项执行审计已通过，精度目标另列未达到。

**精度结论：部分达到，未普遍达到用户参考图或TRAIN PDF水准。** 每个数据集首折三种子确认均未达到90%全种子MAPE通过、80%稳定参考及无>10%失败的联合工作目标。没有继续无限追加轮数。

| 数据集 | 全源电芯 | 正式TEST电芯 | 三种子全部MAPE≤3% | 稳定参考数值通过 | 单种子MAPE≤3% | 平均MAPE | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| MIT | 134 | 134 | 102/134 | 6/134 | 343/402 | 2.186% | 部分达到；未达到普遍参考水准 |
| XJTU | 55 | 55 | 39/55 | 2/55 | 123/165 | 2.508% | 部分达到；未达到普遍参考水准 |
| HUST | 77 | 16 | 8/16 | 0/16 | 29/48 | 2.626% | 部分达到；未达到普遍参考水准 |
| UConn_ISU_ILCC | 64 | 11 | 7/11 | 4/11 | 25/33 | 2.171% | 部分达到；连续形态证据不足 |
| Zhu2022_NCA | 55 | 55 | 38/55 | 14/55 | 136/165 | 2.096% | 部分达到；未达到普遍参考水准 |

稳定参考数值通过要求全部三种子MAPE≤3%、至少两个种子满足全程/后期MAPE及起点/终点/阶段可评价指标；不能用均值曲线替代逐种子验收。该列不等于PDF式完美拟合。UConn仅有稀疏RPT观测点，缺失阶段指标仍为unavailable，连续逐循环曲线尚未证实。HUST和UConn的固定TEST仅16及11颗，其余源电芯参与TRAIN/VAL。不同折的TRAIN/VAL电芯组不能视作独立物理电芯。

## 全名单图集、原始结果与失败名单

每个角色分别提供PNG/SVG/PDF、逐电芯逐种子指标和来源NPZ路径/哈希。黑线为原始真值；蓝线为预测；阴影是三个种子的样本SD，并非置信区间。所有失败和真值尖峰保留，已查看的TEST属于开发留出。

### MIT

[training PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/MIT/completed_015/training/all_cells.pdf) / [validation PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/MIT/completed_015/validation/all_cells.pdf) / [test PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/MIT/completed_015/test/all_cells.pdf)

[training失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/MIT/completed_015/training/reference_failures.csv) / [validation失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/MIT/completed_015/validation/reference_failures.csv) / [test失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/MIT/completed_015/test/reference_failures.csv)

[数据集核验记录、配置及原始预测来源](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/MIT/completed_015/delivery_manifest.json)

### XJTU

[training PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/XJTU/completed_015/training/all_cells.pdf) / [validation PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/XJTU/completed_015/validation/all_cells.pdf) / [test PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/XJTU/completed_015/test/all_cells.pdf)

[training失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/XJTU/completed_015/training/reference_failures.csv) / [validation失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/XJTU/completed_015/validation/reference_failures.csv) / [test失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/XJTU/completed_015/test/reference_failures.csv)

[数据集核验记录、配置及原始预测来源](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/XJTU/completed_015/delivery_manifest.json)

### HUST

[training PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/HUST/completed_003/training/all_cells.pdf) / [validation PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/HUST/completed_003/validation/all_cells.pdf) / [test PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/HUST/completed_003/test/all_cells.pdf)

[training失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/HUST/completed_003/training/reference_failures.csv) / [validation失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/HUST/completed_003/validation/reference_failures.csv) / [test失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/HUST/completed_003/test/reference_failures.csv)

[数据集核验记录、配置及原始预测来源](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/HUST/completed_003/delivery_manifest.json)

### UConn_ISU_ILCC

[training PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/UConn_ISU_ILCC/completed_003/training/all_cells.pdf) / [validation PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/UConn_ISU_ILCC/completed_003/validation/all_cells.pdf) / [test PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/UConn_ISU_ILCC/completed_003/test/all_cells.pdf)

[training失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/UConn_ISU_ILCC/completed_003/training/reference_failures.csv) / [validation失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/UConn_ISU_ILCC/completed_003/validation/reference_failures.csv) / [test失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/UConn_ISU_ILCC/completed_003/test/reference_failures.csv)

[数据集核验记录、配置及原始预测来源](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/UConn_ISU_ILCC/completed_003/delivery_manifest.json)

### Zhu2022_NCA

[training PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/Zhu2022_NCA/completed_015/training/all_cells.pdf) / [validation PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/Zhu2022_NCA/completed_015/validation/all_cells.pdf) / [test PDF](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/Zhu2022_NCA/completed_015/test/all_cells.pdf)

[training失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/Zhu2022_NCA/completed_015/training/reference_failures.csv) / [validation失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/Zhu2022_NCA/completed_015/validation/reference_failures.csv) / [test失败清单](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/Zhu2022_NCA/completed_015/test/reference_failures.csv)

[数据集核验记录、配置及原始预测来源](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/reports/Zhu2022_NCA/completed_015/delivery_manifest.json)

## 优化收益与失败诊断

新检查点规则已经改为最小化实用验证分数S_val，同时约束起点、尾段、阶段及最差20%电芯，避免旧严格通过数优先而选择整体更差的检查点。其正确选择行为和实际检查点证据均已核验；没有用不同预算的历史模型宣称单因素精度收益。

A共享查询读出和B数字rank=4 LoRA共10个同预算对照都未满足≥5%验证收益及保护条件。部分小幅分数收益不足，或最差电芯误差恶化，因此未采用候选。C/D登记的局部噪声触发条件五数据集均不成立，未强行叠加；R0三种子确认已复用，未重复计费。

[10个A/B同预算验证比较及负结果](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/validation_comparisons.md)
 / [TRAIN/VAL差距与失败类型](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/R0_train_validation_gap.md)

图上主要问题：MIT存在严重短寿命起点/下降幅度偏差（b2c1）、过早下降（TEST b2c47均值MAPE22.930%）及许多最终加速缺失；XJTU短寿命3C/RW偏差明显，R3/卫星工况常漏掉末段加速；HUST中后期下降过早或尾段过平；NCA部分CY45较贴近，但CY35退化过早、部分短寿命CY25尾段下降不足；UConn部分稀疏点贴近，其他电芯仍有后期漂移。TRAIN中也有这些问题，单纯提高轮数不构成已证实的解决方案。

## 下一版需要验证的有限改进

以下为待验证假设，不是本版已实施或已证明的根因，也不据TEST调参。下一版需重新登记有限候选和TRAIN/VAL方案：

1. 从TRAIN/VAL检查首100循环可见信号的单位、缺失通道及跨工况归一化对起点和退化速率的影响，保留现有标签/未来隔离检查。不能以真实SOH作起点锚定。
2. 统计TRAIN/VAL速率节点、饱和比例和梯度，比较有限的多尺度速率解码或分段平滑退化率，所有输出仍依赖LLM历史表示；真实线性电芯不强制弯曲。
3. 比较有限的尾段/阶段损失权重；只在TRAIN监督及VAL选择中用真实阶段标签，TEST推理不能读取未来SOH。评价全程误差、最差电芯和种子稳定性，防止只改善终点。
4. 明确仅首100循环可辨识的工况和个体差异，必要时增加训练集可见工况表示或LLM数字适配；避免用未知真实寿命、未来终点或按待测电芯参数拟合。

## 证据入口

[66次模型独立审计](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/completed_v3_evidence_audit.json)
 / [五数据集17折输入隔离审计](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/input_contract_audit/all_dataset_input_isolation_audit.json)
 / [27项行为测试](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/workflow_behavior_tests.json)
 / [完整402+165+48+33+165条TEST逐种子指标](D:/LLM/SMETimes-main-WINDOWS/experiments/llm_curve_v3/20261013_round_robin/formal_progress.json)

模型证据通过只证明使用真实主干、数据边界及结果可重放，不证明预测精度达标。[完整逐项Goal执行审计](D:/LLM/SMETimes-main-WINDOWS/docs/LLM_SMETimes_v3_goal_completion_audit_20261013.md) 已完成。
