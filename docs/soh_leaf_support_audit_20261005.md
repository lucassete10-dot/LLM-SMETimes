# Stage20: XJTU tree leaf-support diagnosis

Audit the verified fivefold shape run 20261005T082211_584949Z, both frozen
global ET control and the sole shape candidate. Load only each plan's training
and validation cells. No new tuning, no outer cells, no accuracy-based exclusions.

For each training model reconstruct exactly its training rows (96 uniformly
selected future observations/cell), train-only imputation and per-cell tail3
weights. Reconstruct predictions from weighted training labels in each forest
leaf. For each validation cell inspect first/last requested cycle and 16
uniformly spaced points on the existing validation curve. The validation endpoint
is an evaluation coordinate, not a lifespan input or selection criterion.

Report global training horizon, known-protocol training horizon, horizon of the
training cells with nonzero prediction weight, weight assigned to cells ending
before the requested cycle, weighted donor-row cycle, and number/effective number
of donor cells. Verify all reconstructed predictions against stored curves and
the estimator within 1e-12. Record adjacent points with identical leaf assignments.
These are explanatory diagnostics, not new goal gates or proof of causality.
Keep every validation cell and save all donor details. A finding of limited
support does not authorize adding validation future labels to training.
