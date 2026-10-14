# Bounded-prefix NASA arm: all outer cells, including failures

Reuse the prior NASA single-cell Python figure structure and style only.
Question: does train-only input bounding improve held-out future SOH curves?
Truth and prediction are saved frozen-test observations after source cycle100.
Do not smooth curves, clip SOH outputs or remove cells. Plot all four cells,
label Bounded-prefix Ridge and public-development LOBO, show MAPE, late-half,
endpoint error. Five identical deterministic repeats share a figure only after
exact equality validation. Save PNG/PDF/SVG and text/collision/alignment checks,
inspect all four panels. These are not new blind-test curves or evidence that
input bounding establishes the correct SOH under out-of-support conditions.
