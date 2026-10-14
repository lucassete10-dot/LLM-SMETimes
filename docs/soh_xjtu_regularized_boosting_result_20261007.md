# Stage102：正则化直接预测baseline对照结果

运行 experiments/xjtu_regularized_boosting/20261007T063931_780782Z。
原31TRAIN/13验证/11外层不加载，seed20261005。原VITsegments/OLS
不变；两固定HGB损失，各global+local，共4新组件、132完整曲线。
没有循环早停划分、没有新增排除、没有更改选模评分。

| 模型 | MAPE % | 后半MAPE % | 末点pp | drop pp |
|---|---:|---:|---:|---:|
| 冻结ExtraTrees | 2.07087 | 3.08399 | 1.91350 | 2.19371 |
| HGB平方损失 | 2.29945 | 2.90965 | 3.54468 | 3.54674 |
| HGB绝对误差损失 | 2.02120 | 2.55832 | 4.38549 | 4.64560 |

absolute_error虽平均/后半MAPE较好，但末点/drop明显恶化，score
4.97814>原3.37147；平方损失score4.41836。expandfalse，不采用。
这里仍是单折开发内层对照，不冒充正式多种子或新盲测成绩。

producerexit0，verifier46523exit0。4fresh模型、TRAINmedian统计、
独立特征、exact全部bin阈值/baseline/200轮boosting节点、132曲线/
指标评分/标签未来Ah污染/家族/坐标/SHA核验通过。3tests通过。
配置/日志/权重/逐芯原始预测完整保留；没有live进程。

后续不追加HGB叶数/轮数搜索，XJTU现有模型仍保留；需要兼顾末期，
不能因平均数字好而采用。下一阶段可转NASA训练侧主要失败模式，
四队列范围不缩减，全部最终曲线/表/Excel与验收仍未完成。
