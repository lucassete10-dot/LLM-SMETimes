# Fixed training-side discrimination experiment

Use only the 39 TRAIN cells of original NCA_fold1. Original five validation and
11 outer cells never loaded. Five fixed grouped-by-cell KFold splits, shuffled
with random_state20261005, repeated estimator seeds20261005..20261009.
Compare four fixed arms with identical ET128/leaf1/quadratic coefficient targets:
known conditions only; existing early VI prefix including conditions; VI source-
cycle segments including conditions; existing prefix with TRAIN coefficient
labels permuted within each known protocol. Same imputation/train-label scaling.

Own fitted quadratic parameters are target supervision only. In each diagnostic
fold model sees only fitting-member parameters, never heldout coefficients.
Heldout true parameters may be recorded to diagnose target error but are not
features or routing. Prefix reads first100 early V/I only; no query capacity,
SOH or lifetime. All39 normal cells retained. Shuffle labels only inside fitting
members and protocol, save exact permutation. Fixed splits shared by all arms.

Save configs/hashes/coefficients/split lists/permutations/models/logs and complete
training/heldout future curve arrays. Report each seed macro MAE/RMSE/MAPE/late/
endpoint/drop and mean ± sample seed SD, plus paired change versus conditions.
These are overlapping training-side OOF diagnostics, not original outer accuracy,
causal proof or a significance test; do not claim that one shuffled comparison
establishes impossibility. No automatic model adoption, no outer use, no new
candidate budget selected based on outer errors. Any improvement needs original
inner validation and full outer checks before adoption. Four-cohort gates intact.
