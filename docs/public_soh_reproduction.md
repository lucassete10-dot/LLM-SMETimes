# 当前公开SOH实验的复现入口

工作目录D:/LLM/SMETimes-main-WINDOWS，使用已有.venv，Windows控制台
加-X utf8。所有路径按这个工作目录解析。原实验不覆盖；训练脚本建立
新的时间戳目录。结果精度比较必须使用冻结划分、相同候选和原始数据。
新时间戳不是新独立数据。绘图重放只检查来源，不训练或修改预测。

## 已保存的权威结果

- MIT/XJTU及旧NASA树：experiments/stm32_compact_trees/runs/20261004T074429_087457Z。
- NASA主Ridge的真实日志复现：experiments/nasa_signal_trajectory/ridge/20261004T165029_391663Z；与原结果的80份权重/预测精确比较见experiments/nasa_reproduction_audit/20261004T165150_046984Z。
- NCA完整5折/5种子两模型：experiments/zhu2022_nca_forecast/20261004T161453_449950Z。
- Oxford独立稀疏诊断：experiments/oxford_sparse_diagnostic/20261004T145450_603603Z。
- 全队列重算：results/public_soh_validation/20261004T163546_715588Z；256不同物理电芯，1555条模型/电芯/种子评测，不能把NCA两模型计成110颗新电芯。

## 可执行核验

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.verify_nasa_ridge_result --run experiments/nasa_signal_trajectory/ridge/20261004T165029_391663Z
.venv/Scripts/python.exe -X utf8 -m scripts.verify_zhu2022_nca_forecast --run experiments/zhu2022_nca_forecast/20261004T161453_449950Z
.venv/Scripts/python.exe -X utf8 -m scripts.verify_oxford_forecast --run experiments/oxford_sparse_diagnostic/20261004T145450_603603Z
.venv/Scripts/python.exe -X utf8 -m scripts.audit_heldout_curve_archives
.venv/Scripts/python.exe -X utf8 -m scripts.index_nca_visual_review
.venv/Scripts/python.exe -X utf8 -m scripts.audit_nca_future_coverage
.venv/Scripts/python.exe -X utf8 -m scripts.summarize_public_soh_validation
.venv/Scripts/python.exe -X utf8 -m unittest tests.test_nca_future_coverage tests.test_nca_shape_audit tests.test_zhu2022_nca_inputs tests.test_public_soh_summary tests.test_nasa_bounded_prefix
```

人工检查索引仅核对已经明确记录的110份人工笔记、图片哈希与抄录指标，
不自动代替人眼检查。旧树绘图函数曾缺历史间接依赖快照，本次保存当前
重放依赖且70张PNG逐像素一致，不冒称恢复缺失历史快照。

## 在现有准备数据上重训

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.train_nasa_signal_trajectory --arm ridge
.venv/Scripts/python.exe -X utf8 -m scripts.train_zhu2022_nca --smoke
.venv/Scripts/python.exe -X utf8 -m scripts.train_zhu2022_nca
.venv/Scripts/python.exe -X utf8 -m scripts.train_stm32_compact_trees
.venv/Scripts/python.exe -X utf8 -m scripts.train_oxford_diagnostic_forecast
```

MIT/XJTU脚本名保留stm32历史命名，但当前验收不要求上板或量化。
NCA训练读取固定准备目录data/interim/zhu2022_nca/20261004T161215_905258Z，
不是早期错误容量标签版本。需要重建原始准备数据时运行：

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.prepare_zhu2022_nca --raw-dir data/raw/voltage_relaxation/20261004T152932_222520Z
```

重建会生成新的准备目录；先比较审计及哈希、显式设置训练PREP，不能
偷偷改用最新目录改变数据合同。原始包MD5/SHA、来源和完整容量协议
规则见原始intake_manifest及准备audit.json。

## 失败记录不能删除

NASA旧树约7.05%、信号趋势外推10.55%、有界前缀6.202%均保留。
对应最终两失败分别在experiments/nasa_signal_trajectory/20261004T143359_280186Z、
experiments/nasa_bounded_prefix/20261004T154319_160380Z。失败的图和
验证证据见docs/soh_execution_progress_20261005.md。有标签泄漏或
错误数据管线的旧结果不当作有效模型能力，保留失败原因而非清除文件。

当前最终档案审计见docs/public_soh_acceptance_audit_20261005.md；
历史缺日志的事实不能以事后笔记冒充真实训练控制台日志。

## 最终输入隔离与档案检查

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.audit_core_forecast_input_isolation
.venv/Scripts/python.exe -X utf8 -m scripts.audit_public_soh_archives
```

已完成的证据分别在experiments/core_input_isolation/20261004T171851_837342Z
及experiments/public_soh_archive_audit/20261004T173146_681119Z。
前者核验193芯前缀、965次测试预测，不会重训；后者核验当前7个实验
1060份权重与配套文件，不以文件齐全代替科学准确性核验。

NASA信号外推和有界前缀缺日志的历史运行已真实同参数复现；分别使用
experiments/nasa_signal_trajectory/signal_trajectory/20261004T171958_017920Z
及experiments/nasa_bounded_prefix/20261004T172001_284366Z。
交互项仅验证复现为experiments/nasa_interaction_ridge/20261004T172145_185139Z。
运行scripts.compare_nasa_repeat时用--old/--new显式指定两次目录，外层运行
另传--curves原4芯图目录。比较所有候选、最终权重和预测相同，不将这些
日志复现计为新的独立实验；详细路径见最终验收审计。
