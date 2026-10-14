# MIT分段特征小规模结果：保留原模型

run experiments/mit_segment_features/20261006T141941_849519Z；固定
MIT_fold1/seed20261005，83 TRAIN、24内层validation，27外层未加载。
6新候选V/VI/VIT×basic/segments，每候选global+2local，实际18组件。
含旧对照7个候选共749条完整train/validation曲线保存。

输入审计feature_inventory.json：107芯同A123 APR18650M1A LFP/graphite
1.1Ah家族；每芯前100源循环98–100观测，选中V/VI/VIT通道有限值
比例全部1，inf0。V20列、VI35列、VIT49列，排除部分Ah/Wh38/39；
无batch onehot，无新增剔除。协议去掉-newstructure存储后缀，保留
真实物理协议字段。TRAIN能建立局部模型的两协议各7/6芯，其余回退
同家族模型。当前不是任意未知型号自动分类能力验证。

| 内层指标 | 冻结旧赢家 | V分段新候选 |
|---|---:|---:|
| MAPE % | 1.57971 | 1.52487 |
| 后半段MAPE % | 2.66538 | 2.55468 |
| 末点pp | 4.54351 | 4.28447 |
| 下降误差pp | 4.60984 | 4.23523 |
| 起点pp | 0.25098 | 0.21142 |
| 逐芯综合误差均值 | 3.34961 | 3.14981 |
| 逐芯综合误差P90 | 4.33177 | 5.24347 |
| 固定选模score=均值+.25P90 | 4.43255 | 4.46068 |
| 逐芯联合通过 | 16/24 | 18/24 |

V分段虽改善均值和通过数，但高误差尾部变差，原score仍选旧赢家。
其余5新候选score也都高于原模型，expand=false。不临时更换score
或取消P90惩罚来宣称赢，不扩大全折/5种子，不更新正式外层成绩。
单折单种子描述不能作5种子正式结论。

独立77852handleexit0：18新组件fresh refit，独立早期feature/权重/
TRAIN填补/协议支持、exact forests、749曲线及全部指标/原score
和扩展判断、查询容量/未来/AhWh污染/家族拒绝均passed。
冻结旧对照为31叶Histogram Gradient Boosting，不错误按ExtraTrees
解释；核对其输入填补、所有预测树节点/类别位图、bin映射、baseline
及相关推理元数据和原始SHA，重放全部保存曲线，无新baseline训练。
proof为run内independent_verification.json。22相关单元测试passed。
validation_score_decomposition.json解释均值改善但score未赢的原因，
仅描述，不变更模型选择。

本轮确认：扩大到V/I/T全部统计特征不是必然更好，仍需稳健泛化
验证。V分段的平均改善是训练侧积极信号，不是已达标证据。后续
候选先固定预算并保留强旧对照，不能针对已看过的外层失败芯调参。
四队列原范围/验收标准不变；所有结果Excel和完整视觉复核仍待完成。
