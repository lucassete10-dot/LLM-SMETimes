# Stage31: transformed fade-rate HGB smoke

Before training: MIT_fold1, seed 20261005, frozen whole-cell train/validation
only. No outer load. Retain Stage30 training-only ET anchor and imputer,
change ONLY rate regressor to HistGradientBoostingRegressor.
Fixed four candidates: leaves 15/31 x L2 0.1/1; learning_rate=.05,
max_iter=120, min_samples_leaf=20, max_bins=64, early_stopping=False.
Same training targets (anchor-y)/(cycle/100), 96 points/cell and tail weight3.
No clipping, smoothing, query-label calibration or lifetime classification.
Save full training/validation predictions, weights, source/data hashes, logs.
Validate label/future poisoning and hard family rejection.
Select minimum existing selection_score on validation; ties candidate index.
Compare both frozen primary and Stage30 inner-selected blend on identical cells.
Expand to five-fold smoke only if selected score strictly improves both controls,
and mean endpoint/drop errors do not worsen either. One fold is NOT evidence
of final generalization; full goal unchanged. Independent refit/replay required.
