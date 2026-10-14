# Stage24: frozen NASA fixed-reference fade rate, five seeds

Expand the stage23 screened candidate, without additional tuning. Four frozen
whole-cell plans, seeds20261005..20261009, learned_anchor=False, ET128 leaf1
tail3, original float64 rate target and no SOH clipping. Evaluate the fixed
candidate, not a per-outer-cell best-of selector. Complete all20 train-only inner
fits/validation records and write their immutable SHA lock before loading any
outer held-out cell. Then refit the same candidate on each fold train+validation
and evaluate that fold's held-out cell. Validation becomes part of final training
only after the policy is frozen. This is public development CV, not blind testing.

Compare all20 outer curves to the frozen original strong NASA Ridge outer
predictions (five deterministic historical repeats, not five independent
experiments). Preserve original goal gates including NASA4/4 joint pass and
P90 improvements. Save all models, every train/validation/test curve, per-cell
metrics, logs, data/source/config SHA and inner lock. Independently rebuild
training medians/rate targets, refit all40 estimators, replay full predictions,
metrics, pollution invariance, family and membership. Figures remain a separate
required deliverable; numerical success alone cannot complete the full goal.
