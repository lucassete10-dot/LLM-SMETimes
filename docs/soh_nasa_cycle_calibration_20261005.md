# Stage44: NASA cycle-wise signal calibration then forecast

Preregister before training. Four frozen NASA folds, seed20261005 only, original
two train and one validation cell per fold; outer never loaded. This is a smoke
screen, not five-seed/generalization proof. No normal high-error cell exclusion.

Fixed13 candidates: frozen historical inner Ridge winner as control, plus
absolute and delta-to-first5 VI/T modes × Ridge alpha .1/10/100 × last30/60
predicted-health Huber linear trends. Calibration columns5/10/34/35/38/40 are
500s/1000s voltage, prefix mean/std voltage, prefix mean temperature and prefix
temperature change; append known discharge-cutoff voltage. They come solely
from the fixed1000-second waveform prefix, cycles≤100. No capacity, cell ID,
cycle number, actual query SOH or future lifespan enters calibration features.

Join calibration labels only from train cells, source cycles≤100 and positive
finite SOH. Equal contribution per cell, weights scaled to training row count;
train-only median imputer and shifted weighted scaler, float64 SVD Ridge.
Delta mode subtracts each cell's first5 median signal vector, but retains known
cutoff as a physical condition. Both modes learn absolute train SOH, not a
hard-coded initial1. Query past health is predicted from VI/T only. Fit Huber
alpha.01/maxiter500 to these predictions over last30/60 source cycles; extrapolate
in fixed (cycle-100)/100 units, no output clipping or truth-derived query fit.

Score uses the existing mean/full+late+endpoint+drop and .25×P90 criterion.
Tie retains lower candidate index. Expand only if selected score beats frozen
control in≥3/4 folds, pooled mean endpoint/drop no regression and MAPE regression
≤.1pp. Save all fitted models, calibration rows/labels/weights, predicted early
health, full train/validation future curves, source/data/model hashes, logs and
failures. Independently rebuild supervised joins, scaler, Ridge/Huber fits,
future/label poisoning, family gate and arithmetic before any formal expansion.
