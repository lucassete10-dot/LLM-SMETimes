# Preregistered NCA level/change pilot

NCA_fold1: 39 train, 5 validation, seed20261005; 11 outer cells never loaded.
Same frozen inner winner control, same first100-cycle 900s V/I+known-condition
features. All 44 loaded cells have a cycle101 label; do not exclude any normal cell.
Only TRAIN labels supervise a 128-tree, depth8, leaf3 level head at cycle101.
Separate 15-leaf HGB learns y(c)-y(101), same 120/.05/minleaf20/l2=1/bins64.
Two fixed arms: equal-cell tail weights 1 and 3. Level head uses one row/cell.
At query time predict level + delta(c) - delta(101), with no query labels,
capacity, future life, or heldout fitted parameters. No local heads or clipping.

Unchanged original selection_score and guards: new score lower, MAPE regression
<=.1 percentage point, endpoint and decline errors not worse than control.
Save all 132 role curves, two components/arm, source/data/config hashes and logs.
Verify saved-model replay, independent sampling/refits/metrics, label poisoning
and family rejection. Expand only if guards pass; negative trials retained.
This tests supervised changes rather than the previous frozen-rate anchor
correction. Single-fold/single-seed pilot is not final evidence or a blind test.
