# Stage48: frozen outer evaluation of robust three-cell inner selections

Stage47 passed independent156-refit/900-curve audit and fixed expansion screen.
Preregister: inherit its four already saved choices, no additional tuning.
Seed20261005..09, freeze20 selections before any outer load; refit components
on original train+validation three cells. Same frozen four outer cells/baseline.
Stage47 chooses3 level corrections and1 control;25-arm indices map to the
original12 calibration parameters modulo12, no changed alpha/window.

All original goal gates remain. Level correction cannot improve drop-shape
relative to the underlying Ridge; this is a partial diagnostic/adoption test,
not a claim that the goal can be completed by changing only level. Deterministic
five repeats are not five independent stochastic replications.

Save20 bundles/35 component fits and80 full train/test curves, locks/provenance,
per-cell/seed summaries, config/code/data SHA, logs/failures. Independently
rebuild all35 fits and80 curves, verify timing, source identity, input/family
isolation and original gates. Keep all4 cells, no error-based exclusions.
Public already observed outer cells are development CV, not blind evidence.
