# Stage28: within-cell relative early signal anchor, NASA internal validation

Before training, freeze four candidates: all44 channels or original six
partial-signal channels [5,10,34,35,38,40], each Ridge alpha10 or100. Four NASA
frozen whole-cell internal plans, seed20261005. No outer loading or normal-cell
exclusion. Six channels are from the pre-existing early-signal method, not
selected by outer residuals. Reference is median of each query cell's own first
five early signal rows; it uses no capacity/SOH labels. Channels are actual
existing V/I/T-derived measurements, not new measurements.

Training joins TRAIN early raw rows to that TRAIN cell's positive finite SOH
labels, cycle1..100. Features are raw-minus-first5reference, selected44/6 channels;
no cell ID or cycle index. Cell-equal sample weights. Train-only median and
weighted shifted scaler, Ridge SVD. Query anchor is the median of last-three
early row predictions. Existing fixed-reference fade component is unchanged:
p(c)=a_relative+d(c)-d(101). This preserves predicted drop exactly. Same timing
approximation as stage26, no SOH clipping or query-label correction.

Compare to frozen stage26 per-fold selected absolute-anchor models and to
historical strong Ridge inner predictions. Select among four new candidates
only by the fixed inner score. Expand only if >=3/4 folds improve over stage26
selected comparator, mean score/end/drop not worse; also report strong Ridge
comparison. Preserve original goal gates and all four-dataset scope.

Save source/data/config/weight/protocol SHA, all models, training supervision
membership, complete training/validation curves, metrics and actual logs.
Independently rebuild relative references/joins/moments/normal equations and
retrain models, verify full curves, drop invariance, poisoning, family and
membership. One-seed selected internal results are not formal or blind results.
