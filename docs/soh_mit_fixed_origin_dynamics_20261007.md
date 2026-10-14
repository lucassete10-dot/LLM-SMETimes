# Stage83：固定已有起点，单独训练MIT下降形状

训练前固定：MIT_fold1，seed20261005，83训练/24验证，27外层不加载。
旧MIT pace不是起点/形状拆分；NCA level-delta/rate是不同家族HGB方案。
复用Stage82 CurveNet及MIT V/segments特征，不再学习绝对SOH起点。
预测=p_old(101)+net(c)-net(101)。其中p_old是已核验锁定feature blend。
TRAIN监督目标=y(c)-y(101)，不能把查询y101或未来参数作为输入。
这与对Stage82输出事后平移不同：本轮从头用下降量loss训练。
2候选width16/32，400epoch，lr .001/decay .1，≤96点/芯、尾权3、芯等权。
相同训练侧填补/标准化；固定origin，未引入查询容量或真实寿命。
原评分与扩展门不变：评分降低、MAPE回退≤.1pp、末点/drop不增加。
保存全部214曲线、配置/SHA/权重/epoch loss/逐芯指标。独立重拟合、
origin身份、变化公式、单调性、污染/家族与全部评分核验后才可扩展。
原CurveNet的anchor输出在差分中消除，有效动态输出3个；不虚称所有
参数都参与训练。单折单种子，不替代全Goal的5种子全折标准。
