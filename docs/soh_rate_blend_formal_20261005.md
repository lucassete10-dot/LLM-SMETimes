# Stage30：MIT五种子固定全局rate混合正式开发CV

Stage29五折一粒种子筛选通过，非主目标通过。保持五折整芯名单、
种子20261005..20261009、rate比例0/.25/.5/.75/1和原curve score。
每个fold/seed原主模型按冻结train-only验证选择规则选定；rate模型
只用train拟合smooth-knee anchor参数和128棵ET，不按outer调整。
各候选整个未来曲线使用单一比例；最小分数，平局weight0。

全部25个inner选择及组件/预测SHA保存并锁定后才读取outer芯。
outer用train+validation重新拟合原主模型和rate，训练curve参数只
来自train+validation，不含query标签。rate比例复用已锁inner选择。
已知工况/物理家族隔离，query前100源循环V/I/T输入；无真实寿命、
测试容量/SOH、测试曲线拟合参数、未来信号。全部正常芯保留。

比较冻结旧压缩树和当前tail主模型，各自原整芯划分；主Goal所有
数值门槛不变。保存每候选全train/val曲线、outer全train/test曲线、
权重/训练参数/配置/SHA/实际日志/逐芯失败/独立审计。全部来源称
公开开发CV，不是新盲测；只有完整验收通过才宣布Goal完成。
需要独立重建监督、imputer、树和完整预测；先锁再读取outer的
时序亦需独立核验。五种子和低MAPE本身不代表曲线达标。
