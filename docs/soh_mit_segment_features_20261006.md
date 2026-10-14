# MIT早期分段特征：固定预算小规模验证

固定MIT_fold1/seed20261005；83 TRAIN、24内层validation，27外层不加载。
原MIT三批数据均为A123 APR18650M1A LFP/graphite 1.1Ah物理家族，
不与NASA或XJTU混家族。源输入为每循环主4C放电前300秒的51通道，
前100循环有98–100记录（本折train/validation只读审计）。

排除38/39部分Ah/Wh；按名字选V、VI、VIT，分组包含固定窗V积分，
它不是容量标签。每循环取固定窗曲线统计；跨循环比较basic起始5/
末3/差分/10记录实际周期OLS与四个25源循环分段中位数及差分/计数。
不输入batch/group onehot；已知充电协议只去掉-newstructure存储后缀，
两段倍率、切换百分比、诊断倍率等物理字段保留。分组和填补仅TRAIN。
不按真实寿命分类型，测试标签不参与路由。

6候选V/VI/VIT×basic/segments；128 ExtraTrees/leaf1，末期权重3，
各电芯等权，已知协议≥6芯local，否则家族共享。对照为冻结旧inner
选模的tail/reference赢家，不把弱模型当唯一baseline。预计每新候选
1global+2local=3组件，实际以训练支持为准。

原score优先、并列保留对照。只有新候选score更低、MAPE不增加>0.1
百分点、末点与下降误差均不增加，独立核验通过后才考虑5折5种子。
保存输入特征审计、配置/SHA/权重/日志/完整曲线；不新增剔除。
结果仍是小规模内层筛选，不能冒充正式外层成绩或Goal完成。
