# Stage26: signal anchor plus relative fixed-rate trajectory

Before assembly, freeze four NASA internal plans, seed20261005 and the four
stage25 anchor candidates (Ridge1/10/100 and ET leaf5). Do not retrain components
or add candidate search. Use only frozen inner train-only component models.

For a query cell, a=stage25 signal-only pooled anchor; d(c)=stage24 fixed-anchor
rate-model SOH at requested future source cycle c. Return p(c)=a+d(c)-d(101).
101 is a fixed evaluation coordinate immediately after origin100, not the true
lifespan or a query observed label. Anchor is a last-three-early-row estimate,
so this involves a small timing approximation explicitly reported, not an exact
measured SOH101. No smoothing, clipping or query-label calibration.

The change is a constant offset per query curve, so predicted first/last drop
must exactly preserve the fixed-rate component's drop, irrespective of anchor.
Check this invariant on every saved train/validation future curve. Compare
fixed-rate control and frozen strong Ridge inner reference, with identical
selection score. Choose among four new anchors by inner validation only.
Expansion requires >=3/4 scores better than strong inner primary, mean score/
endpoint/drop not worse. Original main goal gates are unchanged.

Save component file/selection/data/source SHA, manifests, all60 training and
validation future curves/metrics and actual assembly logs. Independently replay
components from their original files, explicit scalar algebra, full metrics,
drop preservation, future/label poisoning and family/membership. No new gradient
training is claimed; this is development screening, not formal five-seed success.
