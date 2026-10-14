# NCA quantile-loss pilot, preregistered before fitting

Use only NCA_fold1 training (39 cells) and validation (5 cells), seed 20261005.
Do not load the 11 outer cells. Keep the existing 900-second early V/I prefix,
known conditions, horizon features and equal-cell tail_weight=1 unchanged.
Control is the locked original inner winner. Three candidates: HGB quantile
loss q=0.25, 0.40, 0.50; same 120 iterations, learning_rate .05, 15 leaves,
min_samples_leaf 20, l2=1, max_bins=64, no early stopping. No local heads.

Select with the unchanged inner curve selection_score. Expand only if a new
candidate improves score, MAPE regresses by at most .1 percentage points, and
neither endpoint nor drop error worsens versus control. No outer results enter
selection. This is an exploratory development pilot, not five-seed evidence.
Quantiles are candidate point forecasts, not calibrated confidence intervals.
Save source/config/data hashes, all training and validation curves, weights,
logs and selection. Replay predictions and verify arithmetic separately.

Motivation: verified selected-inner NCA diagnosis showed systematic decline
underprediction; it does not prove the loss function is the cause. Negative
results must be retained. Do not delete failed normal cells or shift curves
using query labels. No modification to previous hashed source files.
