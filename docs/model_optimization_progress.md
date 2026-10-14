# Model optimization evidence — 2026-09-29

Active scope as of 2026-09-30: `docs/revised_soh_research_goal.md` and
`configs/soh_research_v5.json`. Physical specification compatibility, five-fold
whole-cell validation and five training seeds are required for the revised
research. Historical tree results are not evidence of an LLM contribution.

The following findings are historical. The user replaced the active scope on
2026-09-29 with same-type, whole-cell-held-out future SOH forecasting on
MIT/XJTU/NASA, aiming for approximately 6% MAPE. See
`docs/same_type_soh_forecast_goal.md` and `battery_models_v4/same_type_forecast`.
Current-SOH and independent RUL work are deferred under that new scope.

## Findings supported by completed experiments

1. **RUL target expression is the strongest demonstrated bottleneck so far.**
   On the same 43 MIT Batch-2 event cells and five whole-cell folds, the old
   signal-to-absolute-EOL ExtraTrees reproduces its archived predictions.
   Keeping inputs, training rows, weights and hyperparameters unchanged but
   learning `log1p(remaining cycles)` changes RUL MAPE at observed-life
   20/40/60/80% from 17.98/26.03/38.73/63.52% to
   12.27/15.13/14.45/14.94%. This isolates the combined target reformulation;
   it does not separately identify the effect of subtraction and logarithm.
2. **Estimated health helps later RUL predictions.** Nested whole-cell SOH
   cross-fitting gives 12.36/15.10/14.52/11.60% RUL MAPE. At 80%, the gain
   over signal-only log-RUL is 3.34 percentage points (within-cohort paired
   cell-bootstrap interval −4.82 to −1.88). Earlier checkpoint improvements
   are not established. Rare early-failure cell b2c1 is still badly
   overestimated at the early checkpoints; average accuracy is insufficient.
3. **Real-cycle slopes help modestly; elapsed cycles alone do not.** All 36
   XJTU runs (two frozen splits, GRU/TCN, three input arms, three seeds) are
   complete. GRU current-SOH means change from 1.081% to 0.993% in Stage A
   and 2.430% to 2.145% in Stage B after adding elapsed cycles and rates.
   Gains are not uniform across seeds/backbones/horizons. No 0.5% current
   or 1% cross-condition attainment is established. Among the new neural
   arms validation selects TCN-elapsed (A) and TCN-elapsed-rate (B); the
   historical Stage-B tree still has a lower validation composite score.
   These experiments do not justify automatically replacing the selected tree.
4. **A coherent full future curve is now executable on the MIT event cohort.**
   A training-fitted shape, model-estimated current state and predicted RUL
   yield full remaining observed-curve MAPE 2.699%, versus 4.163% for a
   linear curve with the same endpoints and 6.341% for persistence. Fixed
   +10/+50/+100 cycle MAPEs are 0.655/1.675/1.759%. This is a transparent
   endpoint-anchored coupling, not independent validation of a learned
   degradation mechanism. All folds select the upper shape-grid boundary.
5. **Uncertainty remains inadequate.** With 10 calibration cells per fold,
   disjoint from all predictor/shape fitting and test cells, full curve MAPE
   is 2.813%. Per-checkpoint RUL interval coverage is 97.7/93.0/93.0/93.0%,
   but all-four-origin coverage is 86.0%, below the nominal 90%. Median
   widths are 465/306/216/121 cycles. These intervals remain broad.

## Claim boundaries

- MIT RUL and full-curve figures here cover 43 confirmed-event cells, all in
  Batch 2. The 91 right-censored cells retain their original source records;
  they receive no artificial endpoint in these comparisons.
- SOH estimator MAPE 0.646% is restricted to the pre-EOL portion of those
  43 event cells; it is not a four-dataset or all-MIT acceptance metric.
- XJTU has no confirmed 80%-of-own-reference event under the frozen label
  contract; its fixed-horizon forecasts cannot establish project-defined RUL.
- All cohorts have been inspected during development. Blind acceptance and
  new-data validation remain separate work.

## Saved evidence

- `battery_models_v3/xjtu_cycle_time_ablation/README.md`: all 36 matched runs,
  source/data fingerprints, per-seed weights and predictions, full-cell plots.
- `battery_models_v3/mit_health_information_diagnostic/diagnostic.json`:
  original-EOL reproduction, log-RUL target and measured-health diagnostics.
- `battery_models_v3/mit_estimated_health_rul/README.md`: nested SOH/RUL models,
  per-cell predictions, comparisons and conditional bootstrap evidence.
- `battery_models_v3/mit_joint_trajectory/latest.json`: points to the complete
  timestamped trajectory/interval run, including all fold weights and figures.
- The latest run reloaded weights and reproduced all 172 test origins using
  only label-free prefix features. The regression suite has 56 passing tests.

## Remaining work toward the unchanged phase objective

1. Test current-SOH/future-trajectory joint-training conflict with controlled
   current-only versus staged training; use validation to choose the route.
2. Improve condition calibration/health shape where evidence warrants it;
   reuse earlier cross-dataset zero/few-shot results, with calibration cells
   disjoint from test. Do not interpret oracle residual centering as deployment.
3. Compare the coherent curve against the best relevant prior trajectory
   baseline, examine knee/regeneration errors and the rare-failure cases, and
   assess prospective prediction origins and interval width/coverage tradeoffs.
4. Consolidate NASA/CALCE/MIT/XJTU current/future/RUL evidence by task and
   label availability. Do not force an event metric on unobserved endpoints.
5. Deliver a consolidated model/report with explicit passed/missed targets.

Every future completed training run must retain its artifacts; see
`docs/training_artifact_policy.md`. The next experiment should address one
of these unresolved hypotheses rather than repeat the completed timing audit.
