# Stage23: NASA fixed-reference fade-rate internal fourfold

Freeze stage21 learned_anchor=False candidate, ET128 leaf1 tail3, seed20261005.
Use four frozen NASA whole-cell train/validation/test plans from the original
strong Ridge run; only train and validation are loaded per fold. Each train has
two cells, each validation one. This is small development screening, not proof
of reliable generalization or formal five-seed goal acceptance.

No anchor fitting or anchor search: a=1 is a fixed reference prior, not query
capacity. Fit r=(1-y)/(1+h), decode p=1-(1+h)*r. No SOH clipping or exclusions.
Compare each full validation curve with the original strong inner Ridge model
chosen by its already frozen historical selection, not outer train+val refit.
Do not reselect Ridge hyperparameters or count deterministic repeats as new evidence.

Save data/source/config/model/selection SHA, actual logs, train and validation
full curves and metrics. Independently rebuild training medians, weighted rate
targets and ET from the registered floating operation order; audit equivalent
algebra, full curves, metrics, source membership and future/label pollution.
Expand only if at least three of four folds and mean fixed curve score improve,
with no worse mean endpoint/drop errors. Original goal still requires NASA4/4
outer cells and all numerical gates; no gate weakening based on these results.
