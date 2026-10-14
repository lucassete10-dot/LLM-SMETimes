# Stage29：MIT固定全局退化速率混合

动机：stage22速率模型五折验证末点更好，但下降幅度均值略差，未扩大。
不追逐外层失败芯；只使用stage22相同五折train/validation和种子20261005。
固定预算：rate权重0、.25、.5、.75、1；0和1是已完成组件对照，
新增仅三个混合比例。各fold一个固定权重用于全部芯与全部未来时间，
无按芯/真实寿命/未来SOH调权、无曲线clip或平滑。

组件：stage22已独立复核的learned-anchor fade_rate，与冻结原tail主模型
按原inner validation规则选定的train-only权重。名单必须完全一致，
家族强隔离；query只前100源循环V/I/T/已知工况。允许训练标签，禁止
测试容量/未来信号/真实寿命。此次组件复用，不宣称新梯度训练。

每fold按既定curve score选择，精确平局保留weight0。扩大筛选：至少
4/5fold改善且mean score严格更好、mean endpoint/drop均不差于weight0。
保留所有候选，不挑芯，不读outer；未过不扩大、不放宽门槛。
若通过，再独立预登记五种子完整内层选权和外层refit，不能复用此次
验证成绩当正式外层结果。全Goal数值与科学交付门槛不变。

保存组件路径/SHA、配置/源码/数据SHA、完整train/validation预测与指标、
实际执行日志。独立复核来源名单、原组件预测、凸组合、所有指标和
选择最小值；污染与家族门检查。公开已观察结果仍称开发CV。
