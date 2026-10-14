# Stage102：固定两种损失的正则化直接预测器

Stage101仅稳健趋势未改善，保留原VITsegments/OLS，不继续调斜率。
固定唯一架构HistGradientBoostingRegressor，两配置loss=squared_error/
absolute_error。参数固定max_iter200, learning_rate.05,max_leaf_nodes15,
max_depthNone,min_samples_leaf20,l2_regularization10,max_bins64,
early_stopping=False,random_state20261005。不按循环留出早停集。
这不是LLM，不宣称小语言模型贡献。只测试正则化直接预测baseline。

原31TRAIN/13验证/11外层不加载，现有训练采样≤96、整芯等权tail3，
同一VITsegments/真实循环时距/训练侧median imputer/工况≥6local。
冻结candidate6 ExtraTrees作控制，2新候选4global/local组件，132曲线。
所有正常电芯保留，无新增排除，无查询容量/真实寿命/未来输入。
原score与扩展门：score下降、MAPE回退≤.1pp、end/drop均不增。
配置/种子/划分/耗时/源码数据SHA/权重/真实日志/全曲线指标保存。
独立既有特征构造、4fresh模型/填补统计/全部boosting节点/132曲线/
指标评分/查询标签未来Ah污染/家族/坐标/SHA核验通过才可扩展。
单折单种子冒烟不是正式成绩；通过后固定配置全内层多折5种子验证。
