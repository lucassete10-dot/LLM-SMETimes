# MIT后期损失权重小规模验证：执行前锁定

沿用Stage64 MIT_fold1/seed20261005、83 TRAIN和24 validation；27外层电芯
不加载。原Vsegments特征、128 ExtraTrees、leaf1、local支持≥6均不变。
训练每芯仍最多96个均匀索引点、电芯总权重相等，不输入查询容量或寿命。
仅训练标签侧改变点权重：ramp10、ramp30、terminal10三臂。
ramp为原始线性点序权重1到给定值；terminal10为末四分之一源循环点
权重10、其他1。样本归一化到每电芯和为1，再整体均值为1。

对照Stage64冻结已选融合与纯Vsegments。保留同一selection_score；
小规模扩展要求候选score优于已选融合、MAPE不增加超过0.1个百分点、
末点与下降幅度误差都不增加。只测三个固定候选，无事后追加权重。
保存模型、全TRAIN/validation曲线、配置/SHA/日志，后续独立复核；
单折单种子不称正式成绩。正常困难电芯保留，Goal仍四队列原门。
