# Stage21: transformed fade-rate regression

Motivation: stage20 leaf audit does not support global horizon extrapolation or
constant leaf assignment as the main XJTU cause. It is diagnostic, not proof
that shorter-cell donors cause all errors. Test a change in the regression target
with no new measured inputs and no new exclusions.

Four frozen first-fold train/validation plans, seed20261005, outer unloaded.
Two new candidates per family, identical global ET128 leaf1 tail3 training rows
and early100 input matrix, no hyperparameter search. Frozen old ET is control;
compare to current strong primary before any expansion.

Let h=(source-cycle-100)/100. Predict r=(a-y)/(1+h) instead of y, then return
y=a-(1+h)*r. Denominator1+h avoids amplifying noise near cycle101. Allow negative
r (activation); no predicted SOH clipping or monotonicity enforcement.
Candidate1 a=1 (fixed reference prior, not query measured capacity).
Candidate2 predict a from early signals using one training-row per cell and
ET128 leaf1. Targets a are the train-only anchors from the already independently
verified smooth-knee fits, not any query fitted parameter. Rate training uses
the true fitted TRAIN anchors, not predicted anchors; record this two-stage
training/inference discrepancy and both anchor errors for interpretation.

Reuse stage13 SHA-locked training parameter cache. Train-only median imputation
separately for rate and anchor. Save full train/validation predictions, parameters,
source/config/data SHA, actual logs and metrics. Independently recompute each
transformed target, preprocessing and refit, verify complete curves and pollution
invariance/family gate. This is development screening, not five-seed acceptance.
