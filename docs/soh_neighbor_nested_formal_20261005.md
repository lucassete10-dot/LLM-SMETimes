# Stage41 robust inner-CV formal evaluation, recorded before training

Prerequisite Stage40 independent4095curve/15HGBrefit verification and screen.
5 frozen NCA outerfolds x seeds20261005..09. Each inner selection uses only
original39train cells, same 3 whole-cell splits(shuffle seed20261005 fixed for
all model seeds), same fixed HGB15/tail1 baseline and seven reference mixtures.
Original validation and test not loaded in inner. Every inner model freshly
fit per model seed; each independent child verifier must pass. All25 configs,
selections and model hashes locked BEFORE any outer cell load.
Outer refits train+validation, using locked k/fraction and fixed primary params.
No re-selection using outer. Save all inner candidate full curves and outer
train/test curves, models and pure-primary control. Report cell-equal 5seed
mean/sampleSD, goal gates against frozen old baseline, and separate current
strong primary comparison. Deterministic reference repeats not new cells.
Public developmentCV, no new blind-test claim. Independent outer replay and
full numeric audit required; final scientific plots remain part of full goal.
No automatic adoption, no test-error deletions, no relaxed thresholds.
