# Stage27: NASA decoupled anchor five-seed whole-cell development CV

Expand stage26 only after independent replay passed. Four fixed anchor choices
Ridge1/10/100 and ET128 leaf5, fixed-rate ET128 leaf1 tail3, seeds05..09.
Same stage26 formula, fixed101 coordinate and timing approximation. No new
features, labels as query inputs, exclusions, output clipping or extra search.

For each frozen fold/seed train four anchor candidates on TRAIN early label
supervision and one fixed-reference future-rate component on TRAIN future
labels. Select minimum fixed curve score on that fold's validation cell only.
Save all80 candidate bundles and full training/validation predictions, and lock
all20 selections/model/config/prediction SHA before outer cells are loaded.
Then refit selected anchor and rate components on train+validation and predict
held-out whole cells. This is public development CV, not a fresh blind test.

Compare frozen original strong NASA Ridge outer baseline, with original goal
gates and NASA4/4 requirement. Save all20 outer models, full curves, per-cell/
seed metrics, SHA/config/logs and inner selection evidence. Independent refit/
curve/pollution/membership/scaling and arithmetic checks precede adoption.
Figures and complete manual review remain required before full goal completion.
