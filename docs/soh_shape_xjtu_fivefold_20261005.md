# XJTU waveform-shape fixed five-fold validation

Registered after independently verified first-fold smoke, before loading other folds.
Use the existing five XJTU whole-cell plans and seed 20261005. Outer cells remain
unloaded. First fold is a replay, not an independent replication. This is development
validation, not a fresh blind test and not the formal five-seed goal evaluation.

Only one new candidate: the stage18 extra-trees shape model (128 trees, leaf1,
tail-weight3, global, original predictors plus fixed degree3 Legendre summaries).
No extra feature, parameter or loss search. Compare with the frozen candidate0
extra-trees control and separately with the current validation-selected strong
primary. Preserve all normal high-error cells. Same train-only median and fixed
first100 input contract, no query capacity/SOH or future signals.

Save model/config/source/prepared hashes, full validation predictions and metrics,
actual log and elapsed time. Independently replay all models, preprocessing,
waveform coefficients, pollution invariance and metrics. Expand to formal five
seeds only if mean fixed curve-selection score beats the strong primary and at
least four of five folds improve; endpoint and drop errors must not worsen on
average. These are screening rules, not changes to the original goal gates.
