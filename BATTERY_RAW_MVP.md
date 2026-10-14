# NASA 原始 `.mat` 电池 SOH 最小可行性实验

## 数据来源

- 官方数据页：<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- 官方压缩包：<https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip>
- 本地压缩包：`D:\LLM\NASA\raw\NASA_Battery_Data_Set.zip`
- 压缩包 SHA-256：`82302A7DB4FC1B34E0B6676326610438D43B816BDF11A69D1D012A464EF2F92E`
- 使用电池：B0005、B0006、B0007、B0018。

原始 MAT 文件包含充电、放电和阻抗记录。本实验只读取放电记录中的端电压、电流、温度、时间和容量。容量只用于构造标签，不进入模型输入。

## 任务定义与防泄漏

目标是根据每次放电开始后的前 1000 秒数据估计当前循环 SOH：

`SOH = 当前放电容量 / 该电池首次放电容量`

每 100 秒采样一次电压、电流和温度，并添加前缀区间统计量，共 44 个特征。默认使用相对该电池首次放电的特征变化量（delta），以降低电池间初始状态差异。

- 训练：B0005 + B0006
- 验证：B0007
- 测试：B0018
- 容量和完整放电时长均不进入输入或语义提示。
- 标准化器只在训练电池上拟合。
- 测试电池 B0018 完全不参与训练和调参。

## 结果

| 方法 | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| 循环数线性模型 | 0.02085 | 0.02322 | 2.585% | 0.8910 |
| Ridge | 0.01699 | 0.01923 | 2.149% | 0.9253 |
| ExtraTrees | 0.01181 | 0.01295 | 1.439% | 0.9661 |
| **Random Forest** | **0.00821** | **0.01048** | **0.986%** | **0.9778** |
| 数值 SMETimes | 0.17726 | 0.19047 | 22.441% | -6.3331 |
| 语义融合 SMETimes | 0.11289 | 0.12240 | 14.296% | -2.0282 |

结论：NASA 原始曲线的前 1000 秒足以支持跨电池 SOH 估计，树模型已经给出很强的最小可行性结果。当前冻结 Llama 的 SMETimes 迁移版本在 B0007 验证集表现尚可，但在 B0018 上发生明显域偏移，不能把它当作已经验证成功的最终模型。下一步应保留 Random Forest 作为基线，并为 SMETimes 增加直接数值分支、跨电池归一化和留一电池交叉验证。

## 运行

```powershell
.\.venv\Scripts\python.exe battery_raw_mvp.py `
  --raw_root D:\LLM\NASA\raw\extracted `
  --output_dir .\battery_raw_mvp_results_delta `
  --feature_mode delta `
  --epochs 40 --patience 8 --learning_rate 0.0002
```

关键产物：

- `battery_raw_mvp_results_delta/random_forest_soh.joblib`：最佳模型及特征配置。
- `battery_raw_mvp_results_delta/metrics.json`：完整指标和数据划分。
- `battery_raw_mvp_results_delta/predictions.csv`：逐循环预测。
- `battery_raw_mvp_results_delta/extracted_prefix_features_delta.csv`：提取后的特征。
- `battery_raw_mvp_results_delta/b0018_raw_feature_soh.png`：测试集曲线。

