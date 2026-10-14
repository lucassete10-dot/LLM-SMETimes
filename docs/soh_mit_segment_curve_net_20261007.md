# Stage82 MIT分段特征＋原小型单调网络：固定小试

Stage81 TRAIN83芯中65芯满足末期加速描述规则，采样重建平均MAPE
0.02074%，但最大单点损失4.13pp，不能声称每个尖峰保留。
这不足以证明早期信号能辨识每颗未来拐点，亦非实际路由因果证据。

已有monotone_network与PLS joint，不重新发明同一架构。
本轮只更换输入为已验证的MIT V/segments早期特征，复用原CurveNet。
原网络使用conditional_drift.prefix，本轮不同；容量/SOH无查询输入。
固定MIT_fold1原83TRAIN/24验证/27外层隔离，seed20261005。
两个候选width16/32、AdamW lr .001、weight_decay .1、400epoch。
TRAIN-only median imputer/StandardScaler；采样≤96/芯、尾权3/芯等权。
直接未来曲线MSE训练，不把查询真实未来拟合参数输入模型。
不额外工况局部拆分，先检验共享小网络。

对照原MIT feature blend锁定选模结果，不据外层失败芯调参。
原评分降低、MAPE回退≤.1pp、末点/drop不增加才扩展。
保存配置/来源SHA/模型/epoch loss/214完整曲线与逐芯指标。
推理污染/家族隔离/单调性/公式、重新拟合及指标选择核验后再判断。
这是单折单种子小试，不代替正式多种子及全四队列门。
