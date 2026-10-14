# UConn cycle coordinate resolved from author notebook

This supersedes the cumulative-coordinate blocker in `uconn_cycle_coordinate_access_20261008.md`; other preparation issues remain.

Primary evidence: the author's `notebooks/data_overview.ipynb`, cell 3, sorts each cell's RPT records and uses `np.cumsum(cycles)` to construct its aging axis. `num_cycles` is an interval count, not already cumulative. Public source: https://github.com/REIL-UConn/fine-tuning-for-rapid-soh-estimation/blob/main/notebooks/data_overview.ipynb . Retrieved through a pinned git blob from the already archived repository tree; git object SHA1, byte count and SHA256 verified. Notebook not executed. Local `overview_provenance.json` records immutable provenance.

The author plotting code also fills missing interval counts from adjacent weeks and modifies nonmonotone SOH for interpolation. Neither behavior is copied into prediction preparation. No future-week fill, SOH modification, label smoothing, extrapolated labels or 100-times-RPT artificial coordinates are used.

Prepared data: `data/interim/uconn_causal_cycles/20261008T073156_095768Z/index.json`, all 64 cells. Each has known cumulative source cycles, RPT IDs, 606 pulse-voltage columns (three SOCs, charging/discharging signs, 101 time points), raw capacity in Ah and initial-normalized SOH for audit. The 32 cells with missing interval counts at RPT 42 are truncated at the first unknown interval, retaining valid earlier records and the cells themselves. Independent reconstruction from authenticated raw author pickle checked all 64 cell coordinates, capacity labels and signal dimensions.

No measured current or temperature is present in this processed pulse file. They are not fabricated. Coulomb-derived SOC, capacity and SOH are not input features. Pulse protocol current/temperature mapping needs author tracker or protocol metadata; group ID alone is not a physical protocol description for deployment.

Actual RPT observation counts:

- N=10/20/30/50: 64/64 cells have only their initial RPT.
- N=75: 58 cells one RPT; six cells two RPTs.
- N=100: 36 cells one RPT; 28 cells two RPTs.
- N=150: 58 cells two RPTs; six cells three RPTs.
- N=200: 34 cells two RPTs; 24 cells three RPTs; six cells four RPTs.

This is a sparse diagnostic dataset, not continuous V/I recording on every source cycle. A prefix extractor requiring three separate cycles cannot simply be reused, and 10 versus 50 often provides identical actual inputs. Future length selection must report observation counts and coverage rather than manufacture observations by interpolation.

Remaining before training: verify experimental nominal capacity/version and group protocol metadata; implement a sparse-prefix feature contract with absent information represented explicitly. The current manufacturer page lists typical capacity 1.2 Ah (https://lithiumwerks.com/products/lithium-ion-18650-cells/), but does not establish the exact nominal reference used in these older experiments. No training-ready or goal-complete claim is made.
