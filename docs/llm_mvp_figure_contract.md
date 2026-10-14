# MIT MVP curve delivery contract

Question: Does the validation-selected LLM + SMETimes candidate follow each
held-out development cell's observed future SOH trajectory?

One single-panel time-series figure per all 27 eligible held-out cells. Python
matplotlib workflow retained. Source-cycle coordinates map to x; saved initial
comparable-capacity SOH maps to y. Display raw observed labels and predictions
without smoothing, clipping, interpolation or omitted points. Input N=100;
one seed, one frozen development fold. No uncertainty band because no replicate
spread exists. No independent-test or performance-success claim.

Export 8 by 5 inch internal-review PNG at 300 dpi plus editable PDF/SVG. Minimum
glyph size 5 pt. Single-panel alignment not applicable, recorded. Mandatory
font/collision checks for every PDF. Reuse existing typography/layout only;
discard old seed averaging and old row-count tail metric. Report tail by source
cycle span. Main plots show one validation-selected candidate, not per-cell
oracle selection. All failed examples remain in the delivery.
