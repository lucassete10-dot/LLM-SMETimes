# Stage97：个性化形状头原内层直接对照

Stage96独立核验passed。VIT形状头在TRAIN内部相对无特征均值改善
27/31，但最佳MAPE6.477%、末点10.302pp，不能冒充超越冻结模型。
按预定诊断门进入原31TRAIN/13验证。仍仅两个alpha10/100，不新增
参数搜索；形状目标/特征/范围限制均沿用Stage96。不看11外层。
TRAIN逐芯oracle重用已核验缓存；全TRAIN拟合imputer/scaler/Ridge。
冻结原candidate6为控制；3×44完整曲线，2新头，0树重训。
score下降、MAPE回退≤.1pp、end/drop均不增才允许全折扩展。
失败保留全部记录，不再追逐这条独立参数头的alpha。不删难芯。
核验重新拟合2头/全部TRAIN统计，独立公式与132曲线指标评分，
缓存SHA/查询污染/家族检查，正式Goal多种子四队列门不变。
