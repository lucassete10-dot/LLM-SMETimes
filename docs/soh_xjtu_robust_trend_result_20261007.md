# Stage101：稳健斜率单因素对照未改善

运行 experiments/xjtu_robust_trend/20261007T063534_543840Z。
原31TRAIN/13验证/11外层不加载，seed20261005；仅末10记录趋势
OLS→Theil–Sen，VITsegments其他特征/ExtraTrees参数/工况路由不变。
新增global+local共2组件，88完整train/validation曲线保存。

验证原MAPE2.07087%、late3.08399%、end1.91350/drop2.19371pp；
稳健趋势MAPE2.09920%、late3.14799%、end2.15751/drop2.40792pp。
score3.37147→3.59640，expandfalse，不采用不全折，不新增窗口搜索。
趋势估计敏感性不等于噪声因果或预测提升，不因此排除记录/电芯。

producer31063exit0，verifier97145exit0。独立scipy.theilslopes对照
手工pairwise median、其他特征独立构造、2fresh组件 exact森林状态/
填补统计、88曲线全部指标/评分/标签/未来/Ah污染、家族/坐标/SHA
核验passed。3单元测试通过，其他特征不变、查询未来标签无影响。
无live进程，保留原模型。单折单种子筛选不冒充正式四队列成绩。

到此已检查共享校准、独立形状、个体残差、子集集成、趋势稳健化；
这些候选均未胜过冻结模型。停止局部特征/修正旋钮搜索，后续优先
回到直接预测模型的正则化/训练损失对照，并保持原内层/多折验证，
不要为得到漂亮曲线继续追逐同一验证芯的最优修正。
完整Goal门不变，所有正常芯保留，四队列验收与全部交付仍未完成。
