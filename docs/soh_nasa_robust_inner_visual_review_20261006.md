# Stage48 NASA all-four-curve review

Final export: results/nasa_robust_inner_curves/20261005T162212_432323Z.
All four PNGs reviewed separately, labels/legend readable and outside lines.
SVG/PDF editable; glyph floor8pt; automated source/alignment/collision checks
passed. No geometry WARN, no axes clipping or smoothing, all future points.
Blue is candidate, gray dashed frozen Ridge, black observed in every figure.
Five deterministic repetitions are explicitly not a CI; no fabricated band.

| Cell | Scientific observation | Visual QA |
|---|---|---|
| B0005 | Candidate shifts too high across much of curve despite close last point. Not perfect fitting; MAPE4.62%. | Pass |
| B0006 | Candidate underestimates SOH, still large systematic offset; MAPE10.43%, joint failure retained. | Pass |
| B0007 | Level offset greatly reduced, captures smooth trend, not individual regeneration peaks; MAPE.80%. | Pass |
| B0018 | Selected control overlaps old baseline, still too high; MAPE4.39%. | Pass |

All curves have short future coverage: first three68 points/span67 cycles,
B0018 32 points/span31 cycles. Explicitly annotated; not evidence for hundreds
of future cycles. No normal cell exclusion; no model switching based on test.
Goal not achieved; scientific limitations are not plotting defects.

One initial ad-hoc data check incorrectly compared each seed prediction directly
to its displayed five-seed mean. A floating reduction difference1.11e-16 caused
exact-array failure. Corrected auditor reconstructs stack.mean and stack.std
exactly and compares them with saved figure inputs; no tolerance relaxation or
prediction/data changes. Data checks do not replace manual visual review.
