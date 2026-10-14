# Stage22: fixed MIT learned-anchor fade-rate fivefold screen

After the stage21 independent verification and strong-primary inner comparison,
freeze the sole candidate learned_anchor=True. Use existing MIT five whole-cell
plans, seed20261005, ET128 leaf1 tail3 and the stage21 original float64 target
operation order. No new parameter/features/anchor search. First fold is replay,
not independent confirmation. This is development validation, not five-seed
formal acceptance. Outer cells remain unloaded and normal failures stay present.

Recompute smooth-knee anchors from TRAIN future labels only in each fold using
the established derive_prior algorithm. Query anchors are predicted only from
early100 signals. Fit rate and anchor medians on training data only. Preserve
config and data/source hashes, actual logs, parameters/models, full training and
validation predictions/metrics. Audit every fold, cache and estimator refit.

Compare fixed candidate to frozen global ET and to frozen validation-selected
strong primary. Expand only if at least four of five folds improve fixed curve
score, mean score improves, and mean endpoint/drop errors do not worsen.
Original main goal numerical gates, data/input scope and five-seed requirements
remain unchanged. No expansion based on outer scores.
