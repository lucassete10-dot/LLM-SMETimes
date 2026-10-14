# Stage47: NASA three-cell training-side leave-one-cell-out calibration screen

Preregister before execution. Keep the four frozen outer folds. For each fold,
load only its original train+validation three cells, never its outer test cell.
Hold each of these three cells out in turn and freshly fit on the other two.
Keep the inherited control Ridge alpha per parent fold unchanged.

Fixed25 candidates: control;12 direct cycle-calibration forecasts;12 constant
level-corrected control forecasts. Calibration grid remains absolute/delta,
alpha .1/10/100, trend window30/60, exactly Stage44. Seed20261005 only because
these models are deterministic. No new raw signals, measured query capacity,
future inputs, true life or test-derived routing. No extra search budget.

Rank candidates by the original fixed goal score pooled equally over the three
held-out inner cells, not by the easiest cell. Compare the Stage45 single-cell
choice on this same expanded validation, and save leave-one-validation-cell-out
selector disagreement as a stability diagnostic, not new hyperparameter search.
All three heldout cells remain in reports regardless of errors/short coverage.

Expansion screen: score improvement in at least3/4 parent folds; pooled selected
MAPE no more than control+.1pp; endpoint/drop mean no worse than control
(1e-10 arithmetic equality tolerance only). No change to main goal thresholds.
Passing this screen is not formal accuracy or goal completion.

Save156 fresh component models, all900 full train/inner-validation curves,
four metrics+late+endpoint/drop+coverage, selections, configs, code/data SHA,
actual logs, failures. Independently rebuild fits and forecast arithmetic,
verify clean inputs/poisoning/family gates and all source hashes. No outer
evaluation automatically authorized by this screen's result. Public development
CV, not a new blind test; three cells remain a very small validation population.
