# LLM＋SMETimes 电池 SOH 预测项目：代码阅读入口

**2026-10-16 更新：优先阅读[完整信号 SOH 估计项目](../projects/full_signal_soh/README.md)。** 它单独发布已训练的完整信号估计模型、五颗未训练验证电芯结果和推理入口。下面的前100周期预测/v4说明保留作历史研究背景，不能混淆两种任务。

本仓库是 Windows 研究工作区的源码快照。主要研究任务是：利用电芯前100个实际源周期的电压、电流、温度、时间及预先已知工况，经真实预训练 LLM 与 SMETimes 数字/文本融合及专家模块，预测后续 SOH 曲线。

当前采用本地 Llama-3.2-1B-Instruct 主干，进行冻结主干、可训练输入投影、融合、专家和曲线解码实验。输入缺失通道有掩码。SOH、容量、未来标签和真实寿命不得作为目标电芯的预测输入。训练集标签用于监督，验证集用于选择候选和检查点；已查看的测试集结果标为开发评估。

## 建议 ChatGPT 阅读顺序

1. 本文件：理解方法、代码入口和运行资料范围。
2. [持续检查与优化 Goal](LLM_SMETimes_reference_tracking_goal_20261014.md)、[v4 执行计划](LLM_SMETimes_curve_optimization_v4_20261014.md)、[曲线参考补充](LLM_SMETimes_v4_prediction_reference_20261014.md)。
3. [真实 LLM 与专家路径](../battery_smetimes/research_llm.py)、[TRAIN 拟合输入预处理](../battery_smetimes/research_sequences.py)。
4. [固定物理速率节点与解析积分](../battery_smetimes/llm_query_rate.py)、[有界局部恢复](../battery_smetimes/llm_local_recovery.py)、[v4 损失权重](../battery_smetimes/llm_curve_objective_v4.py)。
5. [v4 训练器](../scripts/train_llm_curve_v4.py)、[五数据集串行队列](../scripts/run_llm_curve_v4.py)、[候选筛选和三种子确认](../battery_smetimes/llm_v4_workflow.py)。
6. [重载、查询顺序和标签/未来隔离核验](../scripts/evaluate_llm_curve_repair.py)、[验证集评分](../battery_smetimes/llm_curve_quality_v3.py)、[检查点独立审计](../battery_smetimes/llm_v3_selection_audit.py)。
7. [真实预测图导出](../scripts/report_llm_full_dataset.py)、[v4 行为测试](../tests/test_llm_curve_v4.py)、[已完成 v3 五数据集结果](LLM_SMETimes_v3_five_dataset_results_20261013.md)。

## 当前实验范围

五个数据集：MIT、XJTU、HUST、UConn_ISU_ILCC、Zhu2022_NCA。主验收为逐电芯、逐种子 MAPE≤3%，正式种子为20260930、20260931、20260932。除全程 MAPE 外，还检查尾段、起点、终点、阶段下降和种子稳定性；稀疏 RPT 按真实观测点评价。

v3 的有限训练/评估已完成，精度仅部分达到。v4 正在比较两项独立改动：K 增加固定物理速率节点时间分辨率；L 增强 TRAIN 尾部与阶段损失。单种子筛选有效才进入三种子确认。当前项目不能宣称每颗电芯都已达到参考图水平。

## 本仓库包含什么

包含项目源码、测试、配置、数据来源元信息、研究方案和结果文字报告。原始数据、缓存、`.venv`、LLM 权重、训练权重、大型实验目录、个人申请书和生成的设备发布包保存在原工作区，不随 Git 上传。

研究报告中的 `D:/...`、`C:/...` 链接和配置中的本地文件路径属于原工作区记录；公开仓库里没有这些本地运行产物。它们不能证明读者已获取原始预测、也不是下载链接。重放实验需按数据来源文档取得数据和模型，并重建准备数据、冻结划分及审计资料。

## 环境与运行

环境说明见 [Windows 运行指南](../README.md) 和 [requirements.txt](../requirements.txt)。GPU与PyTorch版本须与硬件匹配。以下是已有工作区的入口命令，运行前需准备配置所引用的本地依赖：

```powershell
.venv/Scripts/python.exe -m unittest tests.test_llm_curve_v4
.venv/Scripts/python.exe -m scripts.run_llm_curve_v4 --root experiments/llm_curve_v4/20261014_round_robin
```

原 SMETimes 通用时间序列模块保留在 `models/`、`layers/`、`exp/`、`data_provider/`；电池研究和当前 LLM 曲线优化主要位于 `battery_smetimes/` 与 `scripts/`。历史方法文件用于追溯实验，不代表全部方法同时用于当前模型。
