# NCA unseen-cell curve contract

Question: do low full-future errors also reflect the observed degradation shape,
including late-life failures, for every eligible cell?

Each single-panel quantitative curve is one unseen physical cell. Include all
55 eligible cells for both Ridge and ExtraTrees; show every valid future source
cycle, without smoothing, interpolation, downsampling, or error-based exclusion.
The ten too-short cells and one ambiguous counter-reset cell remain in the raw
audit, not falsely reported as successfully predicted cells.

Python/matplotlib is the established backend. Inherit only typography, palette
and spacing from existing curve plots; build seed-display semantics anew.
Map saved `cycle` to horizontal source cycle, `truth` to observed SOH ratio, and
all five saved `prediction` arrays to forecasts. Raw observed SOH is black;
individual seeds are pale solid blue, their pointwise mean dashed blue. All
individual seed curves encode seed variation, not a confidence interval. Ridge
has five identical deterministic repetitions, explicitly labelled. Metrics in
titles are the average of five individual per-seed metrics, not the metrics of
the ensemble mean. No test-based model selection or favourable examples.

Known inputs are first100 source cycles' measured V/I and experimental
conditions; no test capacity, SOH, later signals or true lifetime is an input.
Source is verified whole-cell 5-fold public development CV, not a new blind test.
Each seed uses the same frozen fold membership and inner-only selection.

Export each plot at 183x100 mm to editable PDF/SVG and 600-dpi PNG, plus a
300-dpi inspection preview. Use sans-serif 8pt body, 10pt title, >=5pt glyphs.
Run source preflight, single-panel alignment (not applicable), PDF text audit
and collision audit. Manual visual review remains pending until each plot has
actually been inspected; automated audits do not establish curve fidelity.
Preserve source hashes, prediction hashes, fold/seed mapping, per-seed metrics,
source arrays, and every failure. No journal-specific submission claim.
