# NASA held-out development curves

Question: does the full future curve match on every held-out cell, beyond cohort MAPE?
Each figure is one frozen outer-test cell, first 100 source cycles as input.
Show every saved valid future observation without smoothing, clipping or exclusion.
Truth and prediction share exact source-cycle coordinates. Report full MAPE,
late-half MAPE and endpoint absolute SOH percentage points. Label the Ridge model
and unseen-cell development evaluation, not training fit or new blind testing.
Five deterministic repetitions must have identical prediction arrays before
sharing a figure; the manifest maps every fold and seed to its exported figure.
Reuse the existing figure helper structurally; only model/evaluation labels differ.
Export PNG/PDF/SVG, check PDF text/collisions and panel alignment. Visually review
all four unique cells, including failures. No confidence bands fabricated from
deterministic repetitions. Do not infer all-cell success from mean MAPE <=6%.
