# Stage25: pooled early-signal anchor supervision

Fixed NASA four internal whole-cell folds, seed20261005. No outer loading or
normal-cell exclusion. Unlike the earlier early-SOH-to-Huber extrapolator, this
uses supervised future dynamics and a separately pooled early anchor predictor.

Training anchor predictor sees only each TRAIN cell's raw44 V/I/T-derived
features for cycles1..100, joined exactly to that TRAIN cell's positive finite
SOH labels. No cycle index, cell ID, capacity or SOH enters the feature matrix.
Per-cell weights are equal. Reference anchor is median SOH of the last three
matched early observations. Query anchor is median of predictions from its last
three early rows; it never reads query labels. This is an auxiliary step in
future forecasting, not a new real-time detection deliverable.

Fixed budget: Ridge alpha1/10/100 (train median, weighted shifted scaler, SVD)
and ExtraTrees128 leaf5. Same future ET128 leaf1 tail3 predicts
r=(a_train-y_future)/(1+h); output a_pred-(1+h)*r. Training rate uses true TRAIN
early anchors, query uses predicted anchor; preserve this discrepancy in logs.
No output clipping, no future input signals, no fitted query parameters.

Compare against frozen stage24 fixed-anchor inner model and historical strong
Ridge inner model, not their outer refits. Save all models/parameters, training
and validation future curves, anchor supervision membership/statistics, source/
data/config SHA and actual logs. Independently rebuild joins, medians, weighted
scaler/targets and models, check full curves/metrics and poison invariance. Only
inner validation can select candidates; formal expansion requires >=3/4 score
improvements over strong primary, mean score/end/drop not worse. Original goal
gates and entire four-dataset scope remain unchanged.
