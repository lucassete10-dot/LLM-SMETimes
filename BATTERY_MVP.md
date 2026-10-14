# NASA battery SOH MVP

This experiment verifies that the SMETimes pipeline can forecast battery SOH
from the processed NASA Excel workbooks without mixing windows from the same
cell across train and test sets.

## Task

- SOH definition: `Capacity_Ah / first Capacity_Ah of the same cell`
- Input: 32 observed discharge-cycle SOH values
- Output: the next 8 SOH values
- Train cells: B0005 and B0006
- Validation cell: B0007
- Test cell: B0018
- Evaluation: rolling-origin test windows with stride 8

The experiment compares persistence and linear-trend baselines with:

1. numeric SMETimes, using only numerical SOH chunks;
2. semantic SMETimes, fusing each numerical chunk with a frozen-Llama
   embedding of a battery degradation summary.

The source workbooks are read only. Checkpoints, metrics, predictions, prompt
embeddings, and the forecast figure are saved under `battery_mvp_results`.

## Run

```powershell
.\.venv\Scripts\python.exe .\battery_mvp.py `
  --data_root D:\LLM\NASA\数据集 `
  --llm_ckp_dir .\models\llm\llama-3.2-1b-instruct
```

Important limitation: these workbooks contain cycle-level capacity but not raw
voltage/current/temperature traces. This MVP therefore validates SOH sequence
forecasting, not leakage-free estimation of current SOH from electrical and
thermal features.

## Three-seed result

The experiment was repeated with seeds 2026, 2027, and 2028. On the held-out
B0018 cell, the semantic variant achieved mean MAE `0.01637 +/- 0.00069`, mean
RMSE `0.02172 +/- 0.00140`, and mean R2 `0.8591 +/- 0.0182`. The persistence
baseline achieved MAE `0.02120`, RMSE `0.02593`, and R2 `0.7997`.

For the 80% SOH crossing, the observed B0018 crossing was discharge observation
75. The semantic model's rolling-origin crossing estimates were 74, 78, and 76
for the three seeds. This is an online short-horizon threshold estimate, not yet
a long-horizon RUL result.
