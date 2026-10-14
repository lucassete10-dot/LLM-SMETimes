# Stage33: rate loss alignment smoke, preregistered before training

MIT fold1/seed20261005 train/validation only. Keep frozen Stage30 train-only
anchor/imputer and target r=(a-y)/d, d=cycle/100. SOH residual equals
-d*(predicted_rate-target_rate). Thus SOH squared loss requires d^2 weighting
in rate space; existing rate baseline kept unadjusted linear tail weights.
This is a hypothesis for shape improvement, not proof of causality.

Four candidates: denominator power1/2 x ET min_samples_leaf1/3; 128 trees,
seed fixed, n_jobs2. Within each training cell normalize linear(1,3)*d^power
weights to sum1; global rescale to mean1. Equal total cell weight retained.
Only training labels/coordinates determine weights; no query lifespan input.
Power2 aligns SOH loss within a cell up to a cell-specific constant.
Power1 is the fixed softer ablation. No clipping or query-label calibration.
All full train/val forecasts and estimator/source/data evidence saved.
Select existing validation selection_score; ties candidate order. Expand
five-fold smoke only if score strictly better and mean endpoint/drop errors
no worse than BOTH original primary and current selected blend. Otherwise stop
this branch. Independent refit and all-curve replay required. Goal unchanged.
