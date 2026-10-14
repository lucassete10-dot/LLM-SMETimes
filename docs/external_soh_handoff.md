# 新电芯输入与独立评测交接

当前发布候选：`deployment/stm32_release_trees/20261004T083706_476897Z`。输入前100源循环的信号，预测未来SOH；不做实时SOH/RUL。此电脑CLI是参考接口，STM32使用独立C模块，尚未完成实板验收。

## 外部新电芯

先提供真实规格，而非填一个MIT/NASA标签：化学体系、厂家、确切型号、外形、额定容量/电压、电压范围。模板`configs/external_soh_request_template.json`只演示已覆盖A123型号；不能把任意电池照填成此型号。

另提供实际两段充电倍率、切换SOC（0–1）、诊断放电倍率、环境温度。第一版仅支持训练中已有的MIT具体策略和30°C环境（输入舍入容差0.1°C，不是已验证温度泛化）。稳定放电电流的输入检查采用训练实际观测范围加10%守门余量，起始过渡点不与额定容量×倍率硬比；这个余量不是精度保证。

`signals.npz`只能有`cycle`和`raw_features`：源循环整数递增且≤100；MIT按既有逐采样提取器生成51维特征，单位V/A/°C。外部N100请求要求≥90个有效观测且最后观测≥95源循环，防止拿前三个循环冒充前100循环。这个预先声明的外部输入完整性门槛不修改公开队列划分、不删除高误差测试电芯。原始波形转特征参见`deployment/common/README.md`，阶段来自已知充放电协议；没有完整阶段的瞬时V/I不足以预测全部寿命。

不能在信号文件放capacity、soh、未来波形、总寿命/EOL。不要用真实400/800寿命决定类型；未来循环数只是用户请求。

```powershell
.venv/Scripts/python.exe -m scripts.predict_release_soh --package deployment/stm32_release_trees/20261004T083706_476897Z --signals PATH_TO_SIGNALS.npz --request PATH_TO_REQUEST.json --end-cycle 800 --output PATH_TO_PREDICTION.json
```

输出SOH为0–1附近的容量比，不是0–100数值。默认未来策略保持不变；输出包含模型/输入/源代码指纹、规格请求、支持范围和警告。未知规格、工况不覆盖、输入单位/格式错误会拒绝，不能把拒绝叫作模型预测成功。请求超训练范围会警告，不将训练最长寿命当作新电池真实寿命。大step仍包含明确请求的end-cycle。

## XJTU/NASA的边界

XJTU发布资料记录厂家/化学体系/容量等，但确切型号未核实；NASA确切型号及化学体系也未核实。因此暂不能把任意匿名2Ah电池自动分给它们。需要补充与训练实验确为同型号/同实验族的可核验证据后再注册，而非只改dataset/family字符串。

已审计公共电芯可以用`--public-replay-cell`替代`--request`做复现；必须精确匹配审计信号指纹。公共例子全部参与最终权重训练，不能称独立新电芯结果。

原信号改名仍标记公共重用。未知指纹/新ID仅说明“可以安排独立评测”，不证明电芯真的独立；输出independent_new_cell=null，需核实真实序列号/采集来源。已知公共输入则为false。

## 标签只在预测结束后单独评测

预测程序不接收标签路径。另存`labels.npz`，只能有递增`cycle`和0–1尺度`soh`；另给label_info.json，包含同一cell_id、normalization=`initial_comparable_capacity`以及reference_rule：MIT/NASA=`first_comparable`，XJTU=`median_first5_comparable`。不能把额定容量归一化与初始可比容量归一化混算。

```powershell
.venv/Scripts/python.exe -m scripts.evaluate_release_soh --prediction PATH_TO_PREDICTION.json --labels PATH_TO_LABELS.npz --label-info PATH_TO_LABEL_INFO.json --output PATH_TO_EVALUATION.json
```

输出MAE/RMSE/MAPE/R²、后半程/末100已观测循环/末点误差、完整匹配真值/预测、缺失标签和排除原因。只排除非有限或非正真值；没有因误差大剔除样本。无匹配标签不报0误差。百分数或Ah不能冒充SOH比值。单个外部结果不是五种子/多折统计，也不能自动升格为独立泛化结论。

## 当前待用户提供

- 实际STM32型号/开发板及可用采样或充放电控制器阶段事件。
- 未参与公开开发的新电芯信号、规格、协议及来源/序列号；未来容量测试标签随后提供。
- 如果希望新增XJTU/NASA同型号新电芯，提供物理身份对应证据。

硬件≤20ms、实测峰值RAM、真实传感链以及新电芯预测MAPE仍待验收；不能以公共复现或参考链接大小替代这些目标。
