# Training result preservation

User requirement (2026-09-29): save the results of every completed training run.

New experiments use a timestamped run directory. Preserve model weights,
configuration and seed, cell splits, preprocessing, predictions, metrics,
training logs where available, and comparison figures. Save each completed
fold/model immediately, so interruption does not discard earlier results.
Retain unsuccessful experiments alongside successful ones. A `latest.json`
pointer may change; historical run directories must remain available.

The existing 36-run XJTU timing experiment already stores separate seed/arm
weights, training histories, validation/test predictions and metrics under
`battery_models_v3/xjtu_cycle_time_ablation/runs/` with a code/data fingerprint.
The existing MIT models and fold predictions are under
`battery_models_v3/mit_health_information_diagnostic/` and
`battery_models_v3/mit_estimated_health_rul/`.

New joint-trajectory runs are stored under
`battery_models_v3/mit_joint_trajectory/runs/<UTC timestamp>/`.
Each fold stores its training/calibration/test identities, SOH/RUL weights,
shape parameters, calibration residuals, predictions, and metrics before
the next fold starts. Tree estimators have no neural epoch loss history;
record their parameters and held-out metrics instead.
