# Stage43: training-only reference pace alignment

Stage42 independent replay proves anchor translation alone cannot change drop
error. Test a pace change instead, before any new outer evaluation.
Reuse the same verified Stage41 seed20261005 inner models, five parent pools,
three whole-cell inner folds. No original validation/test loaded. No new fitting
or independent five-seed claim. Preserve all normal held-out training cells.

Fixed19 candidates: Stage42's13 primary/raw/anchor candidates, plus6 rate-shape
candidates k1/k3 × weights .25/.5/1. Primary/reference predicted drop between
fixed source cycles101 and201 estimates relative pace, never using query SOH,
future signals, true lifespan, future final cycle or fitted query parameters.
If either predicted drop ≤.001 SOH, use pace1; otherwise clip primary/reference
drop ratio to [.5,2]. Map queried cycle c to101+pace*(c-101), and borrow the
reference decline from its101 anchor while retaining primary(101). No clipping
of SOH, no post-test calibration or per-cell winner selection.

Select by the existing fixed validation score. Compare best19 with best13,
ties favor lower index. Expansion requires strict score gain in≥4/5 parent
pools, mean endpoint/drop no regression, and MAPE regression≤.1pp. Goal gates
unchanged. Save all19 full OOF curves, predicted anchors/rates, early peer IDs/
distances, configs/source/model/data hashes, real logs and independent replay.
If predicted early-horizon pace is uninformative, report failure, not success.
