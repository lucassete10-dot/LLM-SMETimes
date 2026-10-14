# Stage91 固定两个监督模板路由：原内层绝对SOH验证

Stage90 TRAIN留出已核验alpha10/100有有限变化收益；本轮不搜索新参数。
固定原MIT_fold1 TRAIN83/验证24、外层27不加载，seed20261005为冻结
起点来源。复用Stage87 TRAIN83独立knee模板缓存及冻结feature blend起点。
重新拟合两个TRAIN83 pace Ridge(81已观察、2缺pace但仍作donor)，k3。
距离/同协议≥6回退/逆距离权重/切线延续与Stage90一致。
预测=冻结信号起点+TRAIN模板相对101变化；不能输入查询y101、真实pace。
保存2pace头/214完整曲线/全路由/配置数据源码SHA/权重/日志/逐芯指标。
TRAIN重代入与验证分开。对照锁定feature blend1.4533%及原评分。
最小score候选只有评分降低、MAPE回退≤.1pp、末点/drop不增加才扩展。
不为每颗验证芯选模型、不删失败芯；单折单种子开发验证，非Goal完成。
独立TRAIN-only头重拟合、缓存solver来源/名单、公式/路由/污染/家族/
全部指标/评分/SHA核验通过后再决定是否扩展。
