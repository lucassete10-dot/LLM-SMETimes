# Stage46: frozen NASA signal-calibrated-level outer evaluation

Prerequisite Stage45 independent156-curve screen passes4/4. Fixed four frozen
NASA outer folds, seeds20261005..09. Reuse the already verified Stage45 inner
choices and component provenance, lock all20 fold/seed records before any outer
data load. Historical control alpha per seed must equal the screened alpha.
No additional candidate search, test-driven switching or exclusions.

Refit Ridge future predictor and chosen VI/T calibrator on train+validation
only for each of20 records. Evaluate saved full training and test curves using
predicted101 level correction. Same shape means unchanged drop relative to its
control: this alone cannot achieve the original drop-P90 improvement gate.
It is a partial level-correction experiment; full goal is not narrowed or done.

Compare to frozen original Ridge saved predictions, check all original gates,
source/data/model hashes, fixed101 anchor provenance, label/future poisoning,
family gate, selection-lock timing and fresh independent refits/metrics.
Ridge/Huber deterministic five-seed repeats are not independent experimental
replications; do not claim zero generalization uncertainty from zero seed SD.
Public development CV, not blind data. Save failures and all normal cells.
