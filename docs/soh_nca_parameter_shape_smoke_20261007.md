# NCA parameter-shape pilot fixed before predictive training

TRAIN-only representation diagnosis:
experiments/nca_training_shape_capacity/20261007T045643_146316Z.
39 TRAIN cells only, no validation or outer labels loaded; 117 shape fits are
own-label reconstructions, not early-signal forecasts. Linear/quad/knee mean
reconstruction MAPE .5164/.1926/.1527%; do not cite these as model accuracy.

Pilot NCA_fold1: 39 train, 5 validation, seed20261005; 11 outer cells excluded.
Frozen old inner winner control. Four candidates: quadratic or knee parameters,
ExtraTrees128 with leaf1 or3, existing signal-only first100-cycle 900s V/I
prefix and known conditions. Shape parameters/scale fitted from TRAIN only;
imputation TRAIN only. No local heads, query lifespan, query capacity or labels.
Reuse existing constrained quadratic fit and fixed-grid knee fit. Absolute
source-cycle coordinates, not fractions of true lifetime. Do not fit validation
curve parameters or perform query label-based anchoring. All normal cells kept.

Original selection_score unchanged. Expand iff new best has lower score,
MAPE regression <=.1 percentage point and endpoint/drop no worse than control.
Save 220 role curves, 4 new heads, all config/data/source hashes, models and logs.
Require independent refit/replay, formula arithmetic and input isolation checks.
This is a single-fold/seed development pilot, not final five-seed evidence.
