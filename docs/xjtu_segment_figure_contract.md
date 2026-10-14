# XJTU分段特征外层曲线图合同

核心结论：均值有改善，但部分电芯退化幅度仍偏差，不能以低平均MAPE
证明所有电芯都能贴合。全55芯完整展示，不筛选漂亮样例。
形式为每芯单面板趋势图（quantitative grid的逐芯审计页），55页合订PDF。
不声称投稿期刊最终规格；内部科研审计183mm宽，110mm高，所有字≥8pt。
Python/matplotlib沿用已确认后端，结构性复用曲线导出逻辑，不改模型。

映射：源cycle→横轴源循环；truth→Observed SOH(0–1)；5个seed prediction
→均值曲线及±1 sample seed SD带。n=5训练随机种子不是5独立电芯，不是CI。
每芯仅1条原始truth，真实容量回升保留；不插值、不平滑、不删终点。
灰虚线+灰SD为此前tail模型，蓝实线+蓝SD为本阶段内层选择候选。
指标注明seed-mean metrics（先算每seed误差再平均），不是均值曲线的误差。
输入前100源循环V/I及可用T/已知工况，未来全有效观测；各折测试整芯隔离。
公开已观察评测为development CV非盲测；联合未达标状态和短未来覆盖注明。

每芯保存源数组及新旧原始5seed堆栈/均值/SD、源SHA、PNG300dpi、
PDF/SVG可编辑字、alignment、PDF glyph/collision检查；绘图不拟合模型。
独立核验全部源数组/55芯覆盖和所有PDF自动审计后，逐芯视觉检查。
未完成视觉检查须明确pending，自动审计不替代科学判读。
