# Stage36 NCA five-fold inner smoke, fixed before execution

Requires independently verified Stage35 NCA screen. All five frozen NCA folds,
one seed20261005. Same seven candidates (control + k1/3 x .25/.5/1), exact
reference protocol/fallback, extrapolation, preprocessing and selection_score.
No outer cells loaded. Each fold independently uses training references only.
No changes to source Stage35 run or scripts. All predictions/weights/logs saved.
Expansion screen: at least4/5 folds strictly improve validation score over
current frozen primary; mean selected score strictly better; mean endpoint and
drop errors no worse. Equal-fold means for the small-stage decision, per-cell
metrics also saved. No screen-based deletion, no threshold changes.
Independent all-curve replay is required. If pass, proceed five seeds with all
inner choices locked before outer evaluation. Formal all-cohort goal unchanged.
