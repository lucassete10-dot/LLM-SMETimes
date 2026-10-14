# SOH原始波形特征接口

`soh_waveform.c/.h`提供两条等价路径：数组批处理参考接口，以及固定内存逐采样接口。阶段必须由调用者根据充放电控制器状态或已知协议提供，不自动从任意混合波形中猜测阶段。

```c
SohWaveStream stage;
float features[51]; /* MIT51，NASA44；XJTU每阶段23 */
if (soh_wave_stream_init(&stage, 1) != 0) { /* 拒绝 */ }
/* 对一个已识别MIT主放电段调用，时间为相对秒；不传容量/SOH。 */
int rc = soh_wave_stream_push(&stage, seconds, voltage_V, current_A, temperature_C);
/* rc=0接受；rc=1表示窗口结束后或XJTU重复采样被忽略；负数错误。 */
/* 物理阶段结束后调用一次。阶段不完整、顺序错误时不能把输出送入预测器。 */
if (soh_wave_stream_finish(&stage, features) != 0) { /* 拒绝 */ }
```

- family=1：MIT已知主放电段的前300秒。至少3点，末点/已观察时间覆盖至少285秒。末端缺失小段沿用训练定义的端点钳位；温度缺失使温度派生特征统一NaN，交由训练侧固定填补器处理。
- family=2：NASA已知放电段的前1000秒。至少2个窗口内点且时间覆盖1000秒，消费首个跨越1000秒的采样用于边界插值，与训练合同一致。不是允许使用未来循环或容量。温度必须可用。
- family=3：XJTU已知充电或诊断放电阶段的前120秒，各阶段至少90个独立采样时间，覆盖至少118秒。先处理充电、再处理放电，将两段23维拼为46维；阶段选择/诊断协议检查由上层完成。温度必须可用。
- 原始V/I及必要温度必须有限，时间非负、不倒序。MIT/NASA同时间重复采样按最后值做插值；XJTU重复时间保留第一值。API本身不重新排序，调用方必须拒绝倒序实时采样。
- 原始阶段只需一个状态和一条当前采样，状态大小用`soh_wave_stream_state_size()`查询。PC ABI为480字节；编译期保证状态≤1024字节，不能将这个状态大小当作全部应用RAM。
- 调用者负责按源循环计数，只向循环特征前端输入预测原点之前（当前合同≤100源循环）的特征。XJTU非诊断使用循环不能冒充同等容量诊断。
- 错误后需要重新init。finish后不再push；可以重新init处理下一阶段。每个XJTU阶段可顺序复用同一个状态，另存23个充电特征，不需要存全波形。

复现命令：

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_stm32_wave_stream.py -v
.venv/Scripts/python.exe -m scripts.verify_stm32_waveform --stream --package deployment/stm32_compact_trees/20261004T074429_087457Z_075640_064384
.venv/Scripts/python.exe -m scripts.verify_stm32_waveform --all --stream --package deployment/stm32_compact_trees/20261004T074429_087457Z_075640_064384
```

全队列重放将固定包模型同时用于训练/验证/测试电芯的数值对照，**不是**新的泛化准确率实验；准确率仍使用原先严格整芯外折结果。原始数据阶段边界由Python提供，尚不证明真实ADC、充电控制器阶段事件、板上延迟和峰值RAM已验收。
