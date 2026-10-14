# Stage40 all-parentfold training-only nested CV (before execution)

Prerequisite Stage39 independent819curve/refit verification and screen pass.
All5 NCA parentfolds, each uses ONLY original train cells; original validation
and outertest never loaded per parentfold. Parentfold training sets overlap,
so fivefold results are correlated; not five independent datasets.
Same KFold3(seed20261005)/HGB15/tail1/reference protocol+extrapolation/seven
candidate budget as Stage39. Full original training-side OOF curves saved.
Independent supervision/imputer/HGB/ref/reference replay for each parentfold.
Expansion screen: >=4/5 parentfolds selected candidate strictly improves pooled
score over control, equal-parentfold mean score better and mean endpoint/drop
no worse. Also report each parent's >=2/3 subfold win screen unchanged.
Do not omit failed/control-winning folds or change criterion after outputs.
If pass, record formal5seed plan with locked training-side selections before
outer; compare fixed-HGB baseline separately from previously selected primary.
No adoption or full-goal claim based on this screen. Scope all4 cohorts unchanged.
