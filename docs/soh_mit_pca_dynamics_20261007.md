# Stage86 MIT低自由度动态头：固定小试

上一轮1600轮高维小网络TRAIN改善但验证未改善，故不继续加epoch。
复用MIT V/segments早期特征、冻结feature blend起点，TRAIN83拟合
median/StandardScaler/PCA(full solver)，比较4/8主成分两个固定候选。
与旧PLS joint不同：无训练拟合参数监督/PLS，直接下降曲线loss，固定起点。
每个候选仅仿射3输出头，经softplus变换产生非负rate/accel/knee时距，
变化=-rate*(h-.01)-accel*(hinge(h)^2-hinge(.01)^2)，h=(c-100)/100。
固定seed20261005，AdamW lr .001 decay .1，1600轮，≤96点/芯，尾权3。
初始化延用原CurveNet动态bias(0,0,log(expm1(.2)))，无预测SOH裁剪。
PCA不读验证芯、不用真实寿命作输入，缺少同协议训练则共享同家族。
固定MIT_fold1原83TRAIN/24验证，27外层未加载。保存214曲线、权重、
1600loss、参数与PCA来源、config/SHA/日志。原评分/扩展门不变。
只比较预先固定2候选，不能按每颗验证芯另选模型。未胜则保留旧模型。
头参数15/27不含PCA、填补、标准化、冻结树，不能冒称总模型15参数。
独立重拟合/特征统计与PCA/公式/污染家族/全部指标评分核验后才扩展。
