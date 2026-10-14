# NCA分段信号特征：小规模预注册

固定原NCA_fold1、seed20261005：39 TRAIN、5 validation、11外层不加载。
只读取原900s V/I早期信号，前100实际源循环；容量/SOH/未来真寿命不输入。
4固定新候选：V/VI × basic/segments，保留原已知温度/倍率conditions。
basic为早期起止中位数、差分、最后10实际记录源循环OLS趋势；segments
加4×25源循环中位数、相邻差分、记录数量，空段保留NaN训练侧填补。
未来特征沿用h、h²、log1p(h)及prefix×h，不用查询真寿命归一化。
同128 ExtraTrees、leaf1、tail3、电芯等权；工况local专家支持至少6芯，
否则回退同物理家族全局。所有imputer只拟合各自TRAIN。

对照原tail_selection预先锁定的训练侧胜者，加载旧模型重放，不重新调参。
保存5臂所有TRAIN/validation曲线、模型、数据来源与源码SHA、日志。
预先固定score；扩展须新score优于对照，MAPE增加≤0.1pp且末点/drop均
不增加。正常困难电芯保留，单折单种子不称正式或盲测结果。
