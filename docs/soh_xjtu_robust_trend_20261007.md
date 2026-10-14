# Stage101：单因素稳健趋势提取对照

Stage100 TRAIN-only审计已核验，约5.7%趋势对OLS/TS敏感，但这不是
异常证明。固定唯一新候选：VITsegments中末10实际记录OLS改为
pairwise median Theil–Sen斜率，每100真实源循环；其余特征完全不变。
稀疏/缺失保持原规则，<2有限点返回NaN，不合成或排除记录。
原31TRAIN/13验证/11外层不加载，seed20261005；ExtraTrees128，
leaf1/depthNone/tail3，已知工况≥6local，其余family fallback不变。
对照复用原candidate6，新增global/local各1组件，共88完整曲线。
固定1候选，不增加换窗/阈值/叶数/特征组搜索。
评分和扩展门不变：score下降、MAPE回退≤.1pp、end/drop均不增。
保存配置/全部划分/源码数据SHA/日志/模型/完整指标/真实预测。
独立TS用scipy.theilslopes、其他特征既有独立构造，重拟合2组件/
森林状态/填补统计、88曲线/全部指标评分/标签未来Ah污染/家族核验。
失败不采用，成功核验后才固定规则全折5种子；Goal原门不改变。
