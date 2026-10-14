# Stage45: isolate NASA level calibration from future shape

Training-only screen using verified Stage44 models, same four frozen internal
folds/seed20261005. Original test not loaded. Fixed13 candidates: control Ridge
and each of12 existing VI/T-calibration models supplying predicted SOH at fixed
cycle101. Forecast = frozen inner Ridge(c) - frozen inner Ridge(101) + calibrated
VI/T model(101). No query SOH/Q, future signals, first-future-truth anchor or life.
This is a prediction transformation, not new training or five-seed evaluation.

Same goal validation score, ties favor control/lower index. Expansion requires
strict score gain≥3/4 folds, mean endpoint/drop no regression and MAPE regression
≤.1pp. Same-shape drop equality is audited with1e-10pp arithmetic tolerance,
not a relaxed goal threshold. Save full train/validation candidate curves,
fixed101 predictions/offsets, copied fitted component weights and source hashes,
selection, real process log and independent curve/metric/score replay.

This step tests NASA's large training-side offset instability, not a generic
solution for NCA's speed mismatch. Do not claim improved shape from translation.
