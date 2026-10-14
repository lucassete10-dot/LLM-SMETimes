# NCA milestone figure contract

Question: how well does the provisional 100-source-cycle configuration predict each of the 44 held-out development cells?

Single-panel quantitative line plots. Internal review format, 8 by 5 inches, not a journal submission. Observed capacity/rated capacity is black; the current ExtraTrees forecast is blue. No baseline comparison, smoothing, label clipping or removed failures. Source fields `__cycle`, `__truth`, `__prediction` map to source cycle, nominal SOH, nominal forecast. Five seeds share one held-out fold per cell; the center is their pointwise mean and the band is one sample seed SD, not a confidence or prediction interval. Caption metrics are averages of the five individual-seed metrics, not metrics of the mean curve.

Scope: development CV, not independent confirmation. N=100 is provisional among tested 10/20/30/50/75/100 lengths. A small MAPE does not establish accurate knee or stage shape. Preserve all 44 cells, including poor tails. PNG/PDF/SVG and JSON point sources are exported. Individual visual review remains pending until performed; automated geometry checks are not scientific acceptance.

Reuse: style-only inheritance from the project's existing Matplotlib curve exporter. New data aggregation and source mapping are explicit and checked; no inherited candidate-selection logic. Final PDF text and collision audits are run for every figure. Single-panel alignment is not applicable, recorded by the alignment helper.
