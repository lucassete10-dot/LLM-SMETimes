# Frozen future-SOH protocol, new NCA cohort

Prepared source: data/interim/zhu2022_nca/20261004T161215_905258Z.
Supersedes the earlier peak-Q-only preparation, which incorrectly treated
short low-SOC diagnostic discharges as full capacity observations. Old smoke
experiments from that preparation are invalid data-pipeline trials, not model
performance. Revised labels require full-charge/cutoff protocol and measured
current integral consistency, not a2500mAh cutoff or prediction-error rule.
Raw source counter includes diagnostic records; preserve source IDs/gaps and
never silently renumber to a full-cycle age or claim100 diagnostic observations.
66 original cells,55 eligible under recorded integrity/input/origin rules;
10 have no future after source100 and1 has ambiguous reset. They remain in
coverage/exclusion reporting, not treated as large-error test exclusions.
No future-life stratification and no chemistry-only pooling with NASA.

Freeze5fold whole-cell KFold shuffle seed20261004; within each development
fold, freeze10% (at least1) randomly chosen validation cells, RNG seed20261004+fold.
No test-cell use for normalization or tuning. Keep all eligible test labels
after source100; training may subsample to96 targets per training cell, with
equal total cell weight. Early prefix uses measured V/I in first900seconds of
observed cycles<=100 and known conditions, not capacity or total cycle duration.

Two separately reported arms, fixed candidates before outer evaluation:
Ridge SVD alpha1/10/100, ExtraTrees64 trees max-depth8 leaf3/6/12.
Design: prefix, horizon/100, squared horizon/100, log1p(horizon/100), prefix
times horizon/100, allowing different future shapes. Scaling and median fill
fit training only; ExtraTrees uses the same filled design. Inner validation
equal-cell MAPE selects parameters, then refit all development cells.
Seeds20261004–08; Ridge deterministic repetitions explicitly labelled.
Smoke1fold/1seed first, then full5fold/5seed. Preserve all configs, plans,
candidate/final weights, validation/test arrays and metrics, run log, failures.
Export all normal-cell test curves and record tail errors before acceptance.
This is future forecasting, not the source paper's current-capacity task.
