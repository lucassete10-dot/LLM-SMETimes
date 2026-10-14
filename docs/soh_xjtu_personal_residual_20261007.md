# Stage98：固定起点与主曲线的早期特征个体残差

Stage97独立形状头失败；保留现有candidate6基础模型。
原31TRAIN/13内层验证/11外层隔离、seed20261005，固定2候选alpha100/1000。
复用Stage92已核验31TRAIN五折OOF。每芯OOF基模型未见过该芯标签。
不用这些残差直接做TRAIN内部head交叉验证：其他donor OOF基模型
可能见过held标签；因此只用原13验证评估整个stack，不能报假嵌套成绩。

残差相对变化r(c)=(y(c)-y_first)-(pOOF(c)-pOOF_first)。
h=(c-101)/100，z=[h,h²/(1+h)]；以z(c)-z(first)拟合每芯两个残差参数。
固定≤96均匀点尾权3轨迹与末点各半；ridge .01朝零；解正规方程后
每参数截在[-.1,.1]。这些是TRAIN监督，不是查询输入。
预测头现有VITsegments，TRAIN-only median imputer、StandardScaler、
Ridge alpha100/1000；无查询容量/寿命/未来参数。共享family头，不切碎工况。
输出参数限定在TRAIN监督范围内。预测pnew=pbase+.04*tanh(z(c)@coef/.04)。
修正绝对幅度≤4SOHpp，cycle101严格为0；c_first>101不声称首点不变。
不裁剪SOH总预测，记录实际范围；修正有界不代表物理范围自动成立。

31残差目标拟合、2新头、0树重训、控制+2候选共132曲线。
全部配置/日志/权重/监督参数/NPZ/指标/源码与依赖SHA保存。
保持原score及扩展门：score改善、MAPE回退≤.1pp、end/drop不增。
独立目标矩阵/线性解、2freshRidge统计、132曲线公式与指标、
修正范围/101锚点/标签未来污染/家族/坐标/SHA核验通过才可能采用。
失败停止继续alpha搜索，正式Goal多种子全部队列门不改变。
