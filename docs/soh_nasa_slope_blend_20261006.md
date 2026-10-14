# Stage49: three-cell inner screen of calibrated-anchor slope blending

Preregister before execution. Reuse Stage47's verified156 component fits and
three-cell rotating inner splits. This is a transformation/selection screen,
not new component training. No outer data loaded. Original25 candidates remain
as controls, add36 candidates: each of12 calibrators with lambda .25/.5/.75.

Blend p=(1-lambda)*level_corrected_Ridge + lambda*direct_calibrated_trend.
Both terms have the same signal-inferred SOH at source101; only future shape
is blended. Thus it can change drop, unlike constant level correction. This is
not earlier tree fade-rate blending, whose inferred-rate learner/input and
anchor are different. No measured query SOH or fitted query labels in either
term, no clipping, lifetime normalization, or outer-error routing.

Select by original goal score over all3 inner heldout cells per parent fold,
same61 candidates each. Compare to Stage47's same-fold locked selected control.
Expand only with score wins>=3/4, selected mean endpoint/drop no worse (1e-10
equality tolerance), MAPE<=control+.1, and a new blended candidate selected in
at least2 folds. Never narrow original gates; this screen is not formal proof.

Save all2196 full train/inner-val curves, four+late/endpoint/drop/coverage
metrics, selected alternatives/jackknife, inherited model and array provenance,
config/code/data SHA/logs/failures. Verify all inherited source SHA/proof,
replay predictions from actual VI/T clean queries and inherited model weights,
family gate/label+future poisoning, independent arithmetic/selection. No new
model fits and deterministic single seed are explicitly labelled.
