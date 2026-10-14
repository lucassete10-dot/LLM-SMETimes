# Stage39 robust train-only inner CV smoke (before execution)

NCA outer-fold1's original39 training cells ONLY. Original5 validation and11
outer test cells never loaded. Seed20261005, KFold3(shuffle=True), whole cells.
Fixed primary HGB15 leaves/lr.05/120iter/minleaf20/L2=1/maxbins64/no early
stop/tail_weight1, selected as a fixed existing baseline before this stage,
not reselected from original validation. This is development work, not fresh
blind validation (historical public results have been seen).
Same seven combinations: control + nearest k1/3 fraction.25/.5/1; protocols,
preprocessing, extrapolation unchanged. Each subfold fits primary and reference
using only its own training cells. All39 held-out predictions pooled cell-equally
for selection_score. Save all training (self-excluded analog) and heldout curves,
weights/config/data+source hashes/logs. Self-excluded training is NOT OOF.
Screen: pooled selected score strictly better than control and pooled endpoint/
drop errors no worse; selected fixed candidate also wins score in at least2/3
subfolds. Independent reference/preprocessing reconstruction and HGB refit
required. If fails, stop this branch; do not force nonzero mixing or alter goal.
If passes, next all5 outer-fold training-side CV then all5seed formal policy,
with all selections locked before outer. No original validation/test calibration.
