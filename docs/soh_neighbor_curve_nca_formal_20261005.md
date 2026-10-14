# Stage37 NCA formal five-fold five-seed evaluation (before execution)

Requires Stage36 independent all-curve and screening verification passed.
All5 frozen folds x seeds20261005..09, identical seven candidates and policy.
All25 inner selections, primary source/choice, components and selection hashes
saved to one lock BEFORE any outer cells are loaded. Candidate ties keep
control/increasing index. No changes from outer outcomes.
Outer refit uses train+validation only for early preprocessing, reference
curves and primary estimator. Test labels used solely to score forecasts.
Reference inputs query early V/I/T/known conditions only; no lifespan warping.
Save full train/val candidate curves and outer train/test curves, all model
bundles, provenance, seed/fold/cell metrics, logs, data/source SHA.
Report full goal gates against frozen old baseline and separate comparison
with current primary. Five-seed variation for deterministic reference component
may be zero; do not treat repeats as new independent cells. Public development
CV, not new blind test. No automatic adoption or final-goal claim from mean
MAPE alone. Independent refits/replays and scientific curve delivery still required.
