# NASA prefix-by-horizon experiment: fixed before new outer predictions

Reason: structure audit proves additive Ridge gives identical future shape for
different development cells within each fold (offset differences only).
This is not a request to pick the best model per outer-test score.

Keep frozen NASA LOBO and single inner validation, origin100 and existing
physical family. Inputs remain early V/I/T and conditions. No new labels or
query-cell capacity input, no renormalization or error-based exclusions.
Add all existing prefix statistics multiplied by horizon/100 and log1p(horizon)
to the existing design. Three fixed Ridge SVD alpha candidates: 0.1/10/100.
Median imputation and weighted scaling fit on training inputs only.
Select alpha solely by inner-validation equal-cell MAPE, then refit development
cells. Preserve every candidate, selection and failures. Smoke uses only fold1
inner validation and never computes outer-test predictions. Full follow-up
uses all4 frozen folds and5 deterministic repeats, exports every test curve,
and reports both additive and interaction arms irrespective of outer outcome.
Do not claim lower errors before measurements; NASA small-sample limitations
and already-observed public-test development status remain.
