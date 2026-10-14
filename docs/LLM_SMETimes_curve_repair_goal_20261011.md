> 最新指令（2026-10-11）：用户已要求先完成 MIT 全数据训练，再训练其他数据集。当前完整队列共 85 次运行；P1 历史失败和最终精度要求不变，旧的“P1 未通过不得启动完整训练”执行顺序由本次用户指令替代。详见 `LLM_SMETimes_full_dataset_execution_20261011.md`。

# LLM + SMETimes 电池 SOH 全曲线优化：诊断与可执行 Goal

日期：2026-10-11。工作目录：`D:\LLM\SMETimes-main-WINDOWS`。

**状态更新：诊断与论文/GitHub 检索已完成，并已开始按本 Goal 执行 P0/P1。严格小样本门槛尚未通过，模型尚未完成修复；P2/P3/P4 未完成。** 原始诊断证据与下一阶段执行规范见本文；新增算法、逐轮训练结果和失败证据见 [执行记录](LLM_SMETimes_curve_repair_execution_20261011.md)。

**执行阶段补充发现：** MIT_fold1 的 83 颗 TRAIN 电芯在第 100 循环的 SOH 范围约为 0.9723–1.0111，而图三 b2c1 为 0.9057；其最后源循环 170 也低于 TRAIN 的最短记录 326。起点和时间跨度均超出本折训练支持范围，支持“状态外推困难”的诊断，但不证明仅此一项导致失败。

单位审计还发现，本地 MIT 特征与统计文本将 `I` 标成 A，而[作者处理代码](https://github.com/chueh-ermon/BMS-autoanalysis/blob/master/cell_analysis.m)把原始安培值除以固定名义 1.1 后存入 `I`。已新增可重放的单位版本转换，将电流及由它派生的短时积分转成物理单位；旧缓存和标签保留。这个常数尺度问题不是已证实的直线塌缩原因，训练标准化会抵消其中大部分数值尺度变化，文本语义仍需纠正。

此外，TRAIN b1c0 含大量局部容量尖峰。对“非负衰减＋单一饱和恢复，总向上变化最多 0.5 pp”的整个曲线族，严格线性规划给出的 raw MAE 下界为 0.23164 pp，高于 P1 的 0.2 pp。需要追查数据一致性与允许的局部恢复表达，不能单靠增加 epoch、按误差删点或放宽验收条件宣称 P1 通过。

新增固定节点 token 查询、完整 TRAIN 标签和 LLM 驱动的有界局部恢复分支已完成小样本训练；局部恢复对照的平均 MAE 为 0.15329 pp，P1 10/12，尚未通过。该结果仅是训练同样本拟合，不是未见电芯预测。另确认一个节点查询检查点的专家路由几乎完全饱和；路由归一化及学习率对照的全部成功/失败记录见执行记录。全部正式分支继续经过真实 LLM+SMETimes，没有用自由答案拟合系数作教师或更改标签。

## 1. 结论：为什么这些曲线接近直线

直接原因已经定位：**当前七参数连续曲线头的拐点参数塌缩到了预测起点附近，模型又没有学准每颗电芯的初始健康状态与衰减速率。** 绘图程序忠实显示了保存预测；没有发现它把非线性输出改成直线。

底层训练原因尚不能归结为单一因素。已有证据支持“端到端欠拟合＋数值参数化饱和＋历史表示压缩”的组合问题；原始信号的个体信息是否足够、数据参考是否有个别错误，仍需后续实验区分。

### 1.1 本次实际复核的来源

- 前四张：`experiments/llm_soh_mvp/20261009T121244_394282Z/llm_numeric_lora4_start_aux/heldout_predictions.npz`，MIT_fold1，输入前 100 个源循环，seed=20260930，选中 epoch=18。
- 实际模型：`battery_smetimes/research_llm.py` 中 `FrozenSMETimesForecaster`、`decode()`、`integrated_soh()`；不是原始 SMETimes 的原样复现，而是其电池回归适配。
- 绘图来源：`scripts/package_llm_current_progress.py`，直接画 `cycle/truth/prediction`。
- 本次独立读取保存权重，对 27 颗留出开发电芯全部重放；与原保存预测的最大 SOH 差为 **0**。
- 本次证据文件：[evidence.json](D:/LLM/SMETimes-main-WINDOWS/outputs/llm_curve_diagnosis_20261011/evidence.json)；复核程序：[audit.py](D:/LLM/SMETimes-main-WINDOWS/outputs/llm_curve_diagnosis_20261011/audit.py)。包含逐芯参数、源文件 SHA256、已有候选验证成绩和第五张图复算。

### 1.2 四张失败图的数值解释

SOH 采用 0–1；1 pp 为 0.01 SOH。以下“起点”是未来第 101 次循环；“转折中心”是输出头内部参数，不是人工检测的真实膝点。

|电芯|实际/预测起点 SOH|实际/预测末点 SOH|预测转折中心（源循环）|最大偏离最佳直线（pp）|主要错误|
|---|---|---|---:|---:|---|
|b1c2|1.00465 / 1.00841|0.96607 / 0.88467|105.04|0.0565|本来缓慢衰减，却持续预测过快；长预测区间累积成大偏差|
|b1c18|0.99834 / 1.01923|0.82646 / 0.80999|100.02|0.1922|把后期加速提前，前中期下降过快，末点接近也不能说明整条曲线正确|
|b2c1|0.90488 / 1.00952|0.77310 / 0.99744|102.65|0.0011|起点高估 10.46 pp，同时严重低估快速退化|
|b2c2|1.00504 / 1.00304|0.77054 / 0.94995|102.04|0.0178|起点基本正确，但遗漏末期加速；末点误差 17.94 pp|

27 颗电芯中 **26 颗**的预测最大偏离最佳直线小于 0.2 pp；预测转折中心中位数为 **103.02** 次循环。这个“直线程度”用于诊断，不能拿来强迫真实线性电芯变弯。

### 1.3 退化成直线的机制

当前输出可写为：

```text
t = (future_source_cycle - origin) / 100
SOH(t) = level - base*t
         - acceleration*width*[softplus((t-knee)/width)-softplus(-knee/width)]
         + bounded_recovery(t)
knee = 5*softplus(raw_knee)
level = 1 + 0.1*raw_level
```

当转折中心靠近起点，后面的积分项逐渐近似线性项；在窄区间内又可能被宽过渡区近似成直线。与此同时，`raw_knee` 很负时 `softplus` 导数接近零，参数难以移回正确位置。b1c18 的 `raw_knee=-10.253`，导数约 3.53e-5；它的过渡宽度约 204 次循环，并非一个定位准确的后期急剧转折。

四个例子的恢复项振幅均接近 **-0.5 pp** 下界，说明这个小幅补偿项也接近饱和。它无法修复 b2c1 的 10 pp 量级起点错误，更无法补足 b2c2 的末期衰减。

**不要只靠降低平滑程度或提高图像分辨率。问题在预测数值本身。也不要认为把所有曲线强制变弯就完成了任务。**

## 2. 已确定的问题、支持证据与待验证假设

|问题|证据等级|依据与含义|
|---|---|---|
|拐点提前且参数饱和|已证实的输出机制|本次直接重放并提取上述参数；是这批近线性预测的直接解释|
|个体起点与速度识别失败|已证实的结果|b2c1 起点及速度均错；b2c2 起点准而后期错，不能用统一平移解决|
|端到端训练欠拟合|已有训练集证据|冻结对照真实 TRAIN 预测仅 2/83 通过联合条件，阶段误差 7.705 pp；不能把失败全部归为测试分布偏移|
|七参数曲线族完全无法表示目标|不受现有证据支持|历史 TRAIN-only 自由参数拟合 83/83 通过，MAE 0.062 pp；但它用了完整 TRAIN 答案，绝不是提前预测成绩|
|专家均衡损失干扰表示学习|有支持，未证明因果|Numeric LoRA 选中权重的固定 TRAIN 小批次中，均衡项在输入投影/路由上的梯度约为预测项的 3.03/4.28 倍；仅一个诊断批次，不能推广为所有训练步骤|
|时间细节被过度压缩|结构上存在，效果待验证|默认每 5 个循环 patch，专家输出再 mean pooling；工况只在头部拼接。具体信息损失须用训练/验证干预测量|
|小范围 LoRA 没解决适配问题|已证实于这一配置|只改最后层 q/v，rank=4；验证结果劣于冻结对照，不代表所有 LoRA 方案无效|
|容量参考/单位/循环拼接错误|尚未排除个别情况|当前标签为初始容量参考；未来毒化检查只证明输入隔离，不能证明原始数据语义正确|
|早期信号不足以唯一决定远期曲线|合理风险，待量化|只看前 100 循环，不可能保证对未知工况、随机事件及不可辨识个体都精确预测；必须通过前缀长度和输入信息实验判断|

训练拟合来源：`experiments/llm_soh_mvp/20261008T133739_668360Z/llm_integrated_start_aux/TRAIN_fit_diagnostic.json`；曲线头表达能力来源：该 run 的 `head_capacity_note.md`、`head_capacity_diagnostic.json`。两者任务不同，不能把 83/83 当作预测准确率。

## 3. 第五张为什么不能直接当成当前 LLM 的成功案例

已按其五个种子原始预测复算，得到 MAPE=0.462238%、Late MAPE=0.692791%、Endpoint=1.174900 pp、Drop=1.632590 pp，与图片一致。

来源为 `experiments/xjtu_segment_outer/20261006T134046_882392Z`，`XJTU_fold3`，电芯 `b1_2C_battery-3`。依据 `choice_lock.json`，前四个种子使用分段 VI 特征 ExtraTrees，第五个种子回退到历史 tail ExtraTrees。实现见 `battery_smetimes/xjtu_segment_features.py` 和 `battery_smetimes/tail_boost.py`。

因此第五张是**历史树模型路线的五种子均值**，不是这版 LLM + SMETimes 的预测。它可以作为曲线贴合目标，不能作为当前架构的已达标证明，也不能把树模型预测混入本轮交付。

此外，旧图的联合阈值是 MAPE≤6%、Late MAPE≤8%、Endpoint≤8 pp、Drop≤5 pp；前四张的联合条件包含 MAE≤1 pp、Tail MAE≤2 pp、起点≤1 pp 等，严格程度和定义不同。新实验必须统一复算，不沿用图上的 Passed 标签作跨方案比较。

“向第五张靠齐”的执行含义：**起点正确、缓慢阶段不过度下降、加速阶段时机与幅度正确、尾段紧跟真实趋势，并且全部电芯公开。** 不要求复制未来测量噪声，也不凭截图外观判定成功。

## 4. 已经试过的改法：避免重复盲调

以下为保存的 MIT_fold1 同一 24 颗验证电芯成绩，主要输入 N=100。单种子开发结果，不是独立确认。

|候选|验证 MAE pp|尾段 MAE pp|阶段误差 pp|联合通过|
|---|---:|---:|---:|---:|
|冻结 LLM 对照＋起点辅助|2.400|4.164|7.831|1/24|
|截图 Numeric LoRA rank4|2.733|5.939|10.862|0/24|
|LoRA 学习率降低十倍|3.048|5.663|7.319|1/24|
|早晚表示差异|2.239|4.601|6.670|0/24|
|统计汇总输入|2.144|4.594|7.288|0/24|
|观测膝点辅助监督|2.609|4.912|7.486|0/24|
|持续加速辅助（10 月 10 日）|2.328|4.607|8.422|1/24|
|固定时间基函数（10 月 10 日）|2.604|5.902|7.659|1/24|

其他历史实验还包括路由归一化、attention pooling、动态差分、融合门控、降低均衡权重、残差专家、局部速度监督，以及 N=50/150/200。均不能直接宣称已解决问题。不同 N 的原始指标因预测区间变化不可直接排序，必须在共同未来区间重算。

截图版本的 27 颗留出开发电芯联合通过 **0/27**，平均 MAE=3.080 pp。冻结对照也为 **0/27**，MAE=2.531 pp。因此恢复对照只是恢复可靠比较基准，不是完成修复。

## 5. 相关论文与 GitHub：可借鉴什么

检索时间：2026-10-11。使用关键词 SMETimes、battery degradation trajectory early prediction、Time-LLM、BatteryMFormer、BatteryGPT、Severson。以下核对到论文入口及作者/官方仓库；没有声称已经在本机复现这些外部模型。网页目录读取有一次失败（BatteryMFormer models 目录），该条仅依据论文与仓库 README，不虚构已逐行审查其模型代码。

|优先级|论文/出处|代码与已核对入口|适合迁入当前路线的内容|限制|
|---|---|---|---|---|
|P0|[Small but Mighty: Enhancing Time Series Forecasting with Lightweight LLMs，2025](https://arxiv.org/abs/2503.03594)|[SMETimes 官方仓库](https://github.com/xiyan1234567/SMETimes)，models、layers、forecasting scripts|对照原方法的数值/语义融合、专家结构和输出解码，记录电池适配究竟改了什么|通用时序实验不是电池全寿命预测保证|
|P1|[Time-LLM: Time Series Forecasting by Reprogramming Large Language Models，ICLR 2024](https://arxiv.org/abs/2310.01728)|[官方仓库](https://github.com/KimMeen/Time-LLM)，[TimeLLM.py](https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py)|在 SMETimes 输入适配中比较数值 patch 重编程与词嵌入对齐，改善直接线性投影进 LLM 的模态适配|不替换 SMETimes 主体；先消融，不认定“加 prompt 必有效”|
|P1|[BatteryMFormer: Multi-level Learning for Battery Degradation Trajectory Forecasting，2026](https://arxiv.org/abs/2605.27044)|[官方仓库，README 标记 KDD 2026](https://github.com/Ruifeng-Tan/BatteryMFormer)，含 aging-condition embeddings、训练/评估入口|借鉴工况条件化解码、早期电信号多视角、退化模式表示，迁为 LLM 后端的可训练查询模块|它不是原样的 LLM+SMETimes；本轮不拿其完整模型替代，也不把测试完整轨迹存为原型|
|P1|[Early prediction of lithium-ion battery degradation with a generative pre-trained transformer，Nature Communications，2025](https://www.nature.com/articles/s41467-025-66819-0)|[BatteryGPT 作者仓库](https://github.com/ReparkHjc/BatteryGPT)，可见 model.py、sample.py、配置及预训练模型入口|借鉴早期充电特征的时序学习、先学习信号动态再学习健康曲线的分阶段训练|GPT 名称不等于自然语言预训练 LLM；其输入与实验协议不能直接照搬，仓库复现完整性需另验|
|P2|[Data-driven prediction of battery cycle life before capacity degradation，Nature Energy，2019](https://doi.org/10.1038/s41560-019-0356-8)|[Braatz 组代码](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)|核对 MIT 数据清理、早期跨循环差分与批次延续关系|主要任务为寿命；ΔQ(V) 使用容量轴，不能悄悄并入“无容量输入”主协议|
|P2|[BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction，2025](https://arxiv.org/abs/2502.18807)|[官方仓库](https://github.com/Ruifeng-Tan/BatteryLife)|借鉴多数据集处理、工况覆盖和生命周期预测的分组评估|寿命标签与完整曲线标签不同；外部数据需查重及隔离|
|P2|Non-destructive degradation pattern decoupling for early battery trajectory prediction via physics-informed learning，2025|[作者仓库与论文说明](https://github.com/terencetaothucb/Early-Battery-Degradation-Prediction-via-Chemical-Process-Inference)|借鉴将初始制造差异、运行条件和退化阶段分别表征，作为 LLM 辅助任务设计依据|不迁入独立化学过程模型或域适配预测器；只借鉴任务分解，收益待验证|

以上是算法设计依据，不是对“每颗电芯都能达到第五张精度”的外部证明。

## 6. Goal：限定 LLM + SMETimes 的实施目标

> 在保留真实预训练 LLM、SMETimes 数值/语义融合与专家学习路径的前提下，修复电池未来 SOH 预测的近线性塌缩、个体起点偏移、拐点错位和尾段误差。先在 MIT 完成可复现修复，再扩展 XJTU 及已有项目范围内的 HUST、UConn–ISU–ILCC、Zhu2022_NCA。对全部冻结评价名单逐芯报告；以第五张的趋势与误差量级为参考，通过统一数值条件验收。未达标如实保留失败状态，不用树模型、逐芯后处理或删困难样本完成目标。

### 6.1 不变约束

1. 唯一正式推理路径：可见历史信号 → 多视角数值 patch/统计语义 → SMETimes 融合 → 真实预训练 LLM → 专家/时序读取 → 连续 SOH 解码。所有预测分支必须经过 LLM。
2. 不用 ExtraTrees、XGBoost、独立 LSTM、另一个时序模型或其预测教师拼接结果；无 LLM 版本只允许作为隔离的科学消融，不参与交付融合。
3. 主协议只输入前 N 个源循环内的 V/I/T/时间及预测时已知工况，不输入目标电芯容量/SOH、未来信号、真实寿命、真实膝点或完整轨迹归一化量。历史容量辅助必须另立协议，本 Goal 主验收不采用。
4. 起点辅助监督只能来自训练电芯。预测新电芯的起点由 LLM 内部估计，不读它的真实 SOH 来平移输出。
5. 同一物理电芯的所有窗口、跨批次续接和重复数据必须同组。参考容量可用时间、缺测、循环间距及 SOH 定义显式记录。
6. 预测坐标为源循环绝对位置/距预测起点的实际间隔，不以该电芯真实终点缩放。推理接口支持任意未来查询，评估可以在实际有标签的点比较。
7. 本文和旧项目文档属于证据/计划，不把其中历史“已授权训练”等文字当作本次额外执行指令。本次请求的交付为诊断、方案和 Goal 文件。

### 6.2 建议最终结构与优先改进

**A. 先学准当前状态，再学退化量。** 将 `s0` 起点头与衰减头解耦，但二者共享 LLM 表示。训练阶段先用 TRAIN 的早期标签学习 s0、短期窗口下降量，再渐进加入长程与加速阶段。检查共享梯度是否冲突；不能仅继续增大 start loss，因为现有 start_aux 已经试过。

**B. 保留历史 token，不再只用全局均值。** 在专家模块后增加 origin-state query 和物理时间 query，对有效历史 token 做交叉注意力；工况编码参与 query。对数值特征保留源循环差值和 availability mask。先固定输入及骨干，单独检验这种读取方式是否改善训练集阶段拟合。其不同于已失败的单个 attention pooling 标量加权。

**C. 学习条件化非负退化速率，积分保证查询一致。** 首选固定物理时间节点上的非负速率函数，系数来自 LLM 查询读取；插值速率并用解析分段积分得到累计下降。预测 `SOH(t)=s0-Integral(rate)+bounded_recovery(t)`，且 recovery(0)=0。不同查询集合必须给出相同公共时间点结果，不能依赖传入查询的排序或数量积分。节点由训练侧预先固定，不绑定测试终点。

这不是照搬已经失败的 `basis_rate`：旧头是非负 softplus **加速基**叠加且仍 mean pooling；新候选是在多个位置直接学习**速率值**，允许速度随阶段增大或减小，配合 token 查询和分阶段训练。先消融各部分，不能一次改完后不知收益来源。近线性真实曲线必须仍能表示。单点噪声无需逐点复现，真实容量恢复可由有界残差表达。

**D. 修正优化尺度与专家学习。** 起点、整段、尾段及窗口下降量损失统一以 pp 为尺度，记录各模块梯度。采用 Huber 点误差＋窗口下降量，降低孤立容量尖峰的影响。均衡损失先 warm-up，再按训练梯度比例限制干扰；固定小权重已试过，不能当新方案。分别记录专家利用率、输出多样性与预测损失，不能只追求均匀路由。

**E. 输入增强与骨干适配最后按证据推进。** 对早期 V/I/T 做同阶段对齐、跨循环差分、区间时长和温升；先验证原始数据具有覆盖，缺失不可插入伪信号。比较输入投影与 Time-LLM 式重编程。只有小样本可拟合但验证泛化差时，再比较最后 2/4 层 LoRA、rank 4/8、分组学习率；保持 text-card 缓存语义正确，文本路径若变动必须失效重算。不要把更大 LLM 作为首要措施。

## 7. 分阶段工作包、产物和门槛

### P0：冻结证据与数据审计

- [x] 对截图来源的 27 颗开发电芯重放，提取 level/base/acceleration/knee/width/recovery；本次完成。
- [x] 核对第五张模型身份和五种子指标；本次完成。
- [x] 建立 `configs/llm_curve_repair_v1.json` 和不可覆盖的实验目录；保存全部数据、代码、预处理器和模型哈希。已通过 P0 源文件与原始数据哈希检查，各训练运行保存独立源码快照及预处理器；当前目录没有 Git 元数据，不使用虚构 commit ID。
- [x] 核对 b2c1 在第 100 次前的原始波形、参考容量、单元身份及批次续接。P0 的 `diagnosis.json`、原始波形与重提取对照已保存，特征重提取误差为 0；没有按误差删除该电芯。
- [ ] 对全部训练电芯统计可见信号、起点标签覆盖、曲线跨度、删失、工况与早期快衰减占比。校验 scaler 仅拟合 TRAIN。
- [ ] 把已反复看过的 MIT/XJTU 标为开发集；确定真正未接触的最终确认名单。重新打乱旧电芯不变成盲测。

产物：`data_contract.json`、`split_manifest.json`、`source_manifest.json`、`diagnosis.json`、`failure_registry.json`。失败数据仍计入覆盖表；只有输入质量规则证明不可用时单列。

### P1：先证明端到端能学会训练样本

- [ ] 从 TRAIN 用预定义覆盖规则选 8–16 颗慢衰减、快速衰减、后期加速、恢复电芯，不选截图中的留出电芯。
- [ ] 运行 tiny-set 过拟合诊断：关闭 dropout/weight decay/均衡项作为诊断对照；仍经过完整预训练 LLM，最多 300 epoch；与恢复常规正则的训练分开存。
- [ ] 为每个模块记录梯度、有效秩、输入/输出方差、LoRA 更新范数及曲线参数分布；检查前后向、mask、尺度和采样。
- [ ] 干预 TRAIN/validation 的早期历史、时间顺序与工况，确认模型对个体信息有响应；“输出改变”仅证明依赖，不证明带来准确率收益。
- [x] 当前曲线头和查询速率头分别在同一冻结 12 颗 TRAIN 上比较，结果在执行记录中逐轮保存。端到端拟合门槛仍未全过，不能据此推进全量训练。

门槛：tiny-set MAE≤0.2 pp、尾段 MAE≤0.5 pp、阶段下降误差≤0.5 pp（真实噪声下限明显更高时记录依据，仍按原阈值标记未过）。过不了就定位实现/优化问题，禁止直接扩成全数据长训练。过了也只证明训练拟合，不证明泛化。

### P2：固定验证集的小规模消融

新增文件已经实现：`battery_smetimes/llm_query_rate.py`、`scripts/train_llm_curve_repair.py`、`scripts/evaluate_llm_curve_repair.py`，以及 LLM 驱动的有界局部恢复分支；复用现有物理类型、预处理、保存及隔离检查。**实现已落地，P2 正式验证消融仍未启动；P1 阶段门槛未通过。**

|实验|相对父配置的唯一主变化|要回答的问题|
|---|---|---|
|E0|重用冻结对照，不重选截图 LoRA|恢复稳定比较基准|
|E1|分阶段训练＋按量纲规范的损失|相同表达结构能否摆脱起点/速度耦合|
|E2|在胜出父配置上增加 token 查询读取|全局均值是否造成关键阶段信息丢失|
|E3|在胜出父配置上替换为条件速率积分头|是否减少 knee 参数饱和，并保留真实阶段差异|
|E4|增加阶段对齐的早期信号/差分 token|可见个体健康信息是否更充分|
|E5|适配方式小搜索：重编程或多层 LoRA|骨干域适配是否有额外收益|

先用相同 seed=20260930 排查，再用 3 个预注册种子复核候选；只推进有验证证据的父配置。主要训练初始设置可沿用 batch_contexts=4、future_targets=96、max_epochs=60；E1 新课程允许 max_epochs=120，patience=12，必须额外提供等更新步数对照，避免把更多训练量误认作结构收益。

未来训练目标采样含均匀覆盖、尾段和持续变化窗口；只由 TRAIN 标签构造，并记录抽样概率/损失权重，避免重复加权。时间导数用实际循环间距和稳健窗口计算，不直接拟合噪声二阶差分。

候选选择仅使用内层验证：先看严格联合通过率，再看按阈值归一化的起点/MAE/尾段/阶段/末点指标；同步检查最差 10% 电芯。一次提交前冻结选择规则。已知 holdout 作为开发诊断报告，不用逐芯选择不同模型。

阶段推进条件：相对 E0，验证 MAE、尾段 MAE、阶段误差均下降至少 20%，起点误差无超过 0.1 pp 的恶化，最差 10% MAE 不恶化；或已达到最终全部逐芯阈值。达不到时保留候选记录、分析失效原因，不强行升级配置。

### P3：历史长度、全名单与多数据集验证

- [ ] 冻结结构后比较 N=50/100/150/200，必要时在达标区间细化；以源循环数而非“保留的第 N 行”为预算。
- [ ] 分别报告共同电芯共同未来区间、各 N 的实际预测区间和全部电芯覆盖。b2c1 这种短寿命电芯不能因 N 增大被悄悄排除。
- [ ] 用验证侧选择最短合格 N；所有 N 不合格则明确“尚无可靠前缀长度”，不能通过延长到接近寿命终点冒充早期预测成功。
- [ ] MIT 全开发折、5 个训练种子；之后 XJTU、HUST、UConn–ISU–ILCC、Zhu2022_NCA 依次推进。物理不兼容体系不直接混训。
- [ ] 推荐最终种子固定为 20260930、20260931、20260932、20260933、20260934（随机整数种子，不解释为日历日期）。
- [ ] 每数据集独立汇总与逐芯验收；小工况组样本不足、无未来标签、无可靠 knee 的情况单列覆盖，不算自动通过。

### P4：统一验收与交付

以下是本方案新提出的严格验收目标，不代表旧模型已满足，也不把第五张单个优秀案例的 0.46% MAPE 强加成所有数据的必然结果。

|单电芯、各次预测的指标|正式目标|
|---|---:|
|未来全区间 MAPE|≤1%|
|未来全区间 MAE|≤1 pp|
|最后 20% 源循环跨度内 MAE|≤1.5 pp|
|首个未来点误差|≤1 pp|
|末点误差|≤2 pp|
|三阶段下降量最大误差|≤2 pp|
|MAE / max(稳健真实下降幅度, 2 pp)|≤25%|
|可信膝点的时间误差|≤预测跨度 10%，无可信膝点记 N/A|
|明显错误平台、持续上升、加速方向反转|不得出现；检测定义在 TRAIN 侧冻结|

评估实现同时保留现有 `curve_metrics` 结果作历史桥接；新条件单独命名，不覆盖旧 `joint_pass`。膝点检测的训练与评价定义必须一致；不能只用分段直线标签监督一个语义不同的平滑参数。

**最终目标覆盖冻结名单的全部可评价电芯，不能用“平均通过”替代全部。** 对每芯五种子均值预测、种子指标均值和最差种子分别报告；严格“全部达标”要求全部可评价 cell×seed 满足指标。若只达到 80%/90% 通过率，只能称阶段性改善。质量不足导致不可评价的电芯仍保留在总名单和覆盖分母中，不据此宣称全覆盖完成。

不存在对未知新电芯百分之百精确预测的保证。若有限输入下仍无法满足全部目标，交付当前真实结果、失败原因、缺失信息和下一步，不伪造曲线、放宽阈值后说完成。

## 8. 自动检查与结果文件

必须实现/保留的检查：

1. 删除所有目标 SOH/label_cycle 并毒化 N 之后信号，预测保持不变；工况字段也检查其可用时间。
2. 单点/批量/乱序/子集未来查询的公共点相同；新增积分头用固定物理节点积分，误差≤1e-5 SOH。
3. 模型重载与保存结果误差≤1e-5 SOH；保存实际 LLM checkpoint 身份和哈希。
4. 线性、加速、恢复、起点偏移、缺失/不规则采样合成例子可由预测头表达；梯度有限，真实线性不被强迫变弯。
5. 窗口和电芯分组无交叉、预处理只训练侧拟合；外部预训练数据不能包含最终确认电芯。
6. 同种子无 LLM/随机骨干/去统计语义消融，仅用于检验贡献，不接入正式推理。骨干扰动改变输出不能单独证明 LLM 有准确率收益。
7. 从逐点 NPZ 独立复算图上所有指标，保留所有失败，不用仅五种子均值遮盖单种子失效。

最终目录要求：

```text
deliveries/LLM_SMETimes_curve_repair_<run_id>/
  README.md
  config.json, data_contract.json, split_manifest.json, source_manifest.json
  experiment_ledger.json, ablations.json, acceptance.json
  checkpoints/                    # 外围参数/LoRA，及骨干哈希/路径
  source_snapshot/
  predictions/<dataset>/<fold>/<seed>/*.npz
  per_cell_metrics.json, worst_cells.json, coverage.json
  figures/<dataset>/<cell>.png
  verification/                   # 重放、未来隔离、查询一致和算术检查
```

每张图采用第五张的信息布局：真实值黑线、预测蓝线、源循环横轴、明确 SOH 参考、输入 N、fold、方法、证据身份与多项指标。五种子阴影只能标“mean ± seed SD，非 CI”；若要预测区间，需要额外做校准并报告覆盖，不能把种子 SD 冒充置信/预测区间。原始逐点值直接绘图，不能依照目标曲线平滑、平移或修剪；训练型趋势输出与真实噪声之间的差异如实显示。

## 9. 可直接运行的诊断命令与下一步启动规范

本次新增审计程序已经运行成功。以下为 PowerShell 命令，不会训练或覆盖旧模型，只重写本次证据 JSON：

```powershell
Set-Location 'D:\LLM\SMETimes-main-WINDOWS'
.\.venv\Scripts\python.exe outputs/llm_curve_diagnosis_20261011/audit.py
```

需要 CUDA 与当前本地 Llama 检查点。预期为 `cells=27`、`max_replay_error=0.0`，并打印四个例子和第五张复算。硬件/依赖改变时仍按 1e-5 SOH 的重放门槛检查。

以下已有入口可供 P0/P1 复用；执行时先复制为新审计工作目录，避免改写既有实验的诊断文件：

```text
scripts/diagnose_llm_training_fit.py --run <run_dir>
scripts/diagnose_llm_head_capacity.py --run <run_dir>
scripts/diagnose_llm_loss_balance.py --run <run_dir>
scripts/verify_llm_soh_mvp.py --run <run_dir>
scripts/verify_llm_curve_metrics.py --run <run_dir>
scripts/train_llm_soh_mvp.py --config <config_file>
```

新架构需要同步扩展 trainer、构造参数、权重加载、验证器及参数清单；不得先给旧 verifier 一个新头，忽略不兼容报错。

执行本 Goal 的任务文本：

> 按本文件 P0→P4 顺序实施，先完成数据语义检查与 tiny-set 可拟合性诊断，再进行有限消融。所有正式预测保留 LLM+SMETimes 路径。每轮根据训练/验证证据选择下一步，记录失败，不以已查看电芯逐芯调参，不用历史树模型替代。达不到门槛时明确指出未完成项；只有覆盖、逐芯精度、模型身份及复现检查全部满足才声明模型 Goal 完成。

完成定义：本轮“诊断与 Goal 文档”已交付；“全名单曲线优化成功”仍待以上工作包执行和验证。
