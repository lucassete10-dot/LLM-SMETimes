# MIT Batch 2/3 跨批次 SOH 验证（2026-09-23）

这是对 `BATTERY_MIT_VALIDATION.md` 中 **Batch 1 已保存模型**的直接外部测试。没有使用 Batch 2/3 标签重训、校准、选特征、调阈值或选 epoch。原始文件从 [MATR 官方站](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)的文件 API 下载；批次命名、续测与坏电池规则参照[作者的 LoadData.m](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m)。

## 数据和协议

| 批次 | 原始电池 | 本次测试电池 | 评估循环数 | 原始 MAT SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Batch 2 (2017-06-30) | 48 | 43 | 20,868 | `63ab200d09ecb237fee5ef3a5c5db76e3212e3206a0bd92f769e1427fed338b8` |
| Batch 3 (2018-04-12) | 46 | 45 | 48,943 | `62c30e413b63e6144720e016deed3661fac8468641794a5807b123fe84717998` |

Batch 2 的 5 个电池（原始零基索引 7、8、9、15、16）是 Batch 1 前 5 个电池的续测，全部排除，避免同一电池跨批次泄漏。Batch 3 的原始零基索引 37（`b3c37`）因作者记录的采集故障从主测试排除。其余按 Batch 1 已固定的规则处理：前 300 秒主放电段、电压/电流/温度，容量仅作为 SOH 标签及质量核对；无效前缀和 >1.25 Ah 明显异常容量予以剔除。评估时各电池跳过前 15 个循环。Batch 2 的 43 个电池有 21,513 个有效循环，测试循环数 20,868；Batch 3 原始 46 个电池有 51,007 个有效循环，排除问题电池并跳过前 15 个循环后为 48,943。

Batch 1 模型曾在其 46 个电池里使用训练 28、验证 8、测试 10；下面 Batch 2/3 结果没有使用 Batch 1 的测试电池重新拟合。指标是**先逐电池求 MAE 再平均**。所有模型都用相同的测试电池和循环。

## 结果

| 冻结的 Batch 1 模型 | Batch 1 内部测试 MAE | Batch 2 MAE | Batch 3 MAE |
| --- | ---: | ---: | ---: |
| 15 维 Ridge | 0.00861 | 0.02512 | 0.00941 |
| 15 维随机森林 | 0.00748 | 0.02326 | 0.00954 |
| 51 维增强 ExtraTrees | **0.00714** | **0.02124** | **0.00870** |
| 15 维数值版 SMETimes | 0.00865 | 0.02573 | 0.00928 |
| 51 维数值版 SMETimes、MSE | 0.00870 | 0.02482 | 0.00895 |
| 51 维数值版 SMETimes、Huber/异常降权 | 0.00867 | 0.02472 | 0.00896 |
| 常数 SOH=1（仅外部批次） | — | 0.04757 | 0.03421 |

MAE 乘 100 即 SOH 百分点：增强树模型在 Batch 2 是 **2.12 个百分点**，Batch 3 是 **0.87 个百分点**。Batch 2 的逐电池平均 R² 为 0.802，Batch 3 为 0.850；与 Batch 1 内部测试相比，Batch 2 明显失准。其逐循环平均有约 **-0.0183 SOH 的低估偏差**，Batch 3 则约 **+0.0079 SOH 的高估偏差**。这支持“批次相关偏移存在”的观察，但不能单凭本实验确定成因。

在 SOH < 0.90 的测试循环里，增强树模型的逐循环 MAE 为 Batch 2 **0.0289**（3,553 循环）、Batch 3 **0.0248**（4,432 循环）。这一阶段误差显著大于全程平均，不能用总体 MAE 宣称寿命末期估计准确。图中 Batch 2 的部分电池还出现预测波动及低 SOH 端的饱和或偏差。

## 结果位置与复现

- `battery_mit_results/crossbatch_frozen_batch1/batch2/metrics.json`、`predictions.csv`、`examples.png`
- `battery_mit_results/crossbatch_frozen_batch1/batch3/metrics.json`、`predictions.csv`、`examples.png`
- `D:\LLM\MIT\raw\batch2_manifest.json` 与 `batch3_manifest.json`：来源、文件大小和 SHA-256
- `D:\LLM\MIT\processed\batch2` 与 `batch3`：逐电池特征、异常审计、清单

```powershell
.\.venv\Scripts\python.exe download_mit_batches.py 2
.\.venv\Scripts\python.exe download_mit_batches.py 3
.\.venv\Scripts\python.exe prepare_mit_crossbatch.py 2
.\.venv\Scripts\python.exe prepare_mit_crossbatch.py 3
.\.venv\Scripts\python.exe battery_mit_crossbatch.py 2
.\.venv\Scripts\python.exe battery_mit_crossbatch.py 3
.\.venv\Scripts\python.exe -m unittest test_mit_pipeline.py test_mit_crossbatch.py
```

本任务验证的是当前循环的 SOH，**不是 RUL**。预处理与指标重算检查共 8 项测试通过。跨批次结果意味着模型在 Batch 3 尚可，但 Batch 2 和低 SOH 区间仍明显不足；在看过外部测试结果后再调整模型，不能把这两批继续视作未触碰的盲测集。
