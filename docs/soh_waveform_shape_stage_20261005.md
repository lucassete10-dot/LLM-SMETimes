# 第十八阶段：既有固定时间窗的波形形状基（运行前登记）

已审计原提取代码：NASA44列=1000s内11点V/I/T+统计量；MIT51列=
300s内11点V/I/T+统计量；XJTU46列=充/放120s内各5点V/I/T+统计量；
NCA32列=900s内11点V/I各加5统计量。现有输入已保留采样波形，不能
声称完全遗漏形状信息。本阶段只改表达方式，不扩大信号窗口或读Q标签。

每条已采样信号在归一化时间[-1,1]拟合Legendre degree3，取非恒定三
系数；>=4有限点可拟合，缺测仍NaN交由训练补值。不加入初始Capacity、
完整充放电时长或true life。仅消除<=64*float64eps*max(1,幅值)的数学
投影舍入误差，避免恒流的零高阶系数被标准化为浮点噪声；不删除样本。
前100源循环内系数汇总initial/last3/delta/全段slope/std/last10slope，
initial与原家族规则一致：NASA/MIT首条，XJTU/NCA前5中位数。
附加到既有query矩阵，不改变已有输入或标签。全部类型硬隔离。

四家族冻结首折seed20261005，只train/validation；控制复用已核验
crossfit_drift/20261005T042311_372306Z的ET128/leaf1/tail3/global。
新候选3：同ET128/leaf1/tail3/global加形状、Ridge alpha10/100加形状，
共12新模型+4冻结控制/16设置。Ridge训练median补值、weighted scaler
先减训练首行再算矩恢复mean，float64 SVD。每训练芯最多96未来标签、
等芯总权重、tail3；不验证早停、不剪裁SOH、不删高误差正常芯。
保存配置/源码数据SHA/模型/真实日志/全部验证曲线和指标。先独立核验
波形契约、训练输入/统计/权重/重训/正规方程、全曲线和污染/家族隔离。
只有超过当前冻结强primary内验证才考虑多折，单折不宣称Goal完成。
