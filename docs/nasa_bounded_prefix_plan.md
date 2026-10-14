# NASA train-only bounded-prefix Ridge

Evidence: frozen inner candidates on two training cells yield validation prefix
z-score maxima118–2855. This is an input-domain extrapolation issue, not a
license to delete difficult normal cells or change their SOH labels.

Before any new outer prediction, freeze candidates: Ridge alpha0.1/10/100,
prefix-z limits none/3/6. Use existing frozen LOBO, origin100,5 deterministic
repeats. Imputation/scaling fit training cells only. Only existing early-prefix
columns (all except final3 requested-cycle columns) may be bounded. Leave
future horizon, truth and output predictions untouched. Training transformed
prefixes must be within the limit, so the fitted estimator sees exactly the
same bounded transform at training and inference. Select all9 candidates by
inner validation MAPE only, refit all development cells, retain every outer
cell and every candidate. None reproduces the original unbounded Ridge.

Report changed-domain column count and all-cell/tail metrics. A clipped signal
is not claimed to measure the correct aging state outside support. Never pick
limits by outer errors, never replace only favourable folds after evaluation.
Start with validation-only execution; if successful, use the predetermined
selection across all4 folds without further search, verify saved weights,
training statistics, label removal and future poisoning, export all4 curves.
