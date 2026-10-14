# Stage42: training-only reference-shape screen

Question: does borrowing a reference's absolute SOH level contaminate an
otherwise useful degradation shape? Use Stage41 independently verified
training-only inner caches, seed20261005, all five parent folds. Each original
39-train-cell pool retains the frozen three whole-cell inner splits. No original
validation or outer cell data/arrays are loaded. Parent pools overlap; 195 OOF
records are not 195 independent batteries. This is a deterministic transformation
screen, not a new five-seed training result or a replacement of the primary model.

Fixed 13 candidates: primary control; raw mixtures k1/k3 × .25/.5/1; shape-only
mixtures with the same k and weights. Shape-only reference at queried cycle c is
primary(101) + reference(c) - reference(101). Anchor is fixed source cycle101,
predicted solely from first100 VI/T and known conditions, never measured query
capacity/SOH, the first available future truth, or future lifespan. Do not warp
time, fit query parameters, clip values, or delete difficult normal cells.

Select solely by the existing fixed validation score (mean of fullMAPE,
lateMAPE, endpointpp, droppp plus .25×cell-scoreP90), ties retain lower candidate
index. Compare the selected winner to the best raw/control candidate under the
same score. Expansion requires strictly better score in ≥4/5 parent pools,
non-increasing mean endpoint and drop error, and mean MAPE regression ≤.1pp.
This is a new pre-registered training-side screen, not revised goal thresholds.
If shape-only adds no value, keep the old candidate and report failure.

Save full OOF predictions for every candidate, anchor values, initial/trend
error decomposition, protocol support/neighbor IDs and early-feature distances,
fixed config/source/data/model hashes and actual logs. Independently reconstruct
anchors, arithmetic, metrics, scores and screen before any further expansion.
