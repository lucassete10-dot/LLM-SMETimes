# 多视角试验结论（Goal v2.1）

固定MIT内层5折、5种子、两配置共50个模型完成。5360条训练/验证曲线逐点重放、训练侧填补与标准化重建、输入未来信号/容量污染隔离及checkpoint排序核验通过。核验不是独立优化器重训。

| 内层设置 | MAPE %（5种子均值±样本SD） | 综合score（不是MAPE） | 参数量 |
|---|---:|---:|---:|
| 已冻结对照 | 1.7817 ± 0.1932 | 5.0855 ± 0.0387 | 原方案，见各折权重 |
| 三视角编码+拼接 | 2.4721 ± 0.1440 | 6.0022 ± 0.2395 | 10577 |
| 三视角编码+门控 | 2.6054 ± 0.2631 | 6.9148 ± 0.2378 | 10724 |

两候选均没有降低预登记综合score。因此不采用、不扩展外层、不追加本轮配置。这里是内层checkpoint选择结果，不是最终未见电芯测试结果。没有证据支持“加入门控即可让所有电芯贴合”。

每种子先在折内电芯等权，再在5折间等权；表中SD不是置信区间。两新模型整条曲线CPU推理中位数约1.68/1.77 ms，仅代表核验环境，不代表STM32运行能力。25次拼接训练累计约99.95秒，25次门控训练累计约120.36秒，不含数据读取、核验和报告时间。

可追溯实验目录：`experiments/multiview_pilot/20261007T110239_771724Z`。

- `config.json`：固定划分、种子、输入视角、源码快照SHA。
- `independent_input_verification.json`：50模型、5360曲线核验。
- `verified_comparison_summary.json`：完整逐种子结果、参数量、耗时、采用判断。
- `verified_comparison_cell_metrics.json`：冻结对照及两候选逐折/种子/电芯内层指标。
- 各折/种子/配置目录：权重、训练历史、训练/验证逐点NPZ、指标。
- `verified_comparison_report.md`：详细比较报告。

生产阶段summary中`independent_verification_pending=true`是核验前历史状态，不覆盖；核验后的权威状态以上述独立核验文件和verified comparison为准。失败的首次冒烟也保留，不把错误运行混入统计。

本候选仅受论文方向启发；未取得完整原文公式，不声称精确复现。阅读边界见`docs/literature/iee_multiview_20261007/paper-card.md`，设计见训练前登记`docs/soh_multiview_pilot_registration_20261007.md`。

多视角有限验证已经完成，但四队列最终Excel、全部逐芯曲线及逐项完成核验仍待交付。Goal保持active。
