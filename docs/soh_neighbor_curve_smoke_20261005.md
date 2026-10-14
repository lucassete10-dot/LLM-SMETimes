# Stage35 nearest-curve reference smoke (before execution)

XJTU and NCA frozen fold1, seed20261005. Train/validation only; no outer.
Fit imputer and StandardScaler to training early prefix features only.
Physical families hard-separated. Match known exact protocol if at least3
eligible training peers, otherwise whole-family fallback. Select nearest k=1/3
by early prefix Euclidean distance; ties cell ID. Equal average of selected
training curves, linearly interpolated on absolute source cycle. Outside
reference support, extend slope OLS on first/last20 observations, anchored at
first/last observed point. No horizon normalization by query lifespan, no
clipping, SOH calibration or prediction-time labels.
Six candidates: k1/3 x reference fraction .25/.5/1 with frozen validation-selected
strong primary. Also save fraction0 control. Training analog evaluation excludes
self (preprocess still fit on all train; not called OOF unbiased performance).
No residual learner this round: test reference component before adding it.
Save every full train/val curve, models, config/source/data SHA, logs/metrics.
Select same fixed selection_score, ties control then increasing fraction/index.
Expand fivefold only if selected score strictly improves control and endpoint/
drop error means no worse. Independent reconstruction/replay required before
expansion. This does not replace final five-seed/all-fold goal.
