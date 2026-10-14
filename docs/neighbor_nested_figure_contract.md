# Nested-CV candidate curve comparison

Question: does the locked training-only nested-CV mixture improve unseen-cell
future curves relative to the current retained model? The conclusion is not
assumed; all 55 NCA outer cells appear, including failures and regressions.
Each figure is a single-panel quantitative trend, not a selected best example.

Structural adaptation of the existing Python/matplotlib curve export workflow:
new predictions now come from outer/test_predictions.npz; the plotted reference
is current_primary with the same five seeds, not the older frozen goal baseline.
The fixed goal gates against the older baseline remain in the verified experiment
summary and are not redefined by this visual comparison. No adoption is implied.

Cycle maps to source cycle; truth maps to evaluation SOH (0–1); prediction maps
to five-seed mean with one sample seed SD (ddof=1), not a CI or prediction band.
Header errors are means of five per-seed errors, not errors of the mean curve.
No smoothing, interpolation, clipping to SOH=1, downsampling or cell exclusion.
Known first 100 source cycles are the only signal inputs. These are development
CV results on previously observed public data, not a new blind evaluation.

Export 183×100 mm, editable PDF/SVG and 600-dpi PNG plus 300-dpi preview,
glyphs ≥5pt. Black truth, gray dashed retained model, blue candidate; all
model curves have the same SD definition. Legend outside the plot. Quantitative
NPZ sources, original prediction hashes, plotting source, configuration and
figure contract retained. No p-values or causal claims. Single-panel alignment
N/A and PDF glyph/collision audits must run after final render. Every cell's
visual review is initially pending until its complete panel is inspected.
The fixed HGB candidate is deterministic here: five seeds yield identical
predictions, so its seed-SD band is zero-width. These are not five independent
replications. Current-model seed variation is descriptive, not calibrated
uncertainty. PDF/SVG are authoritative line-figure exports; PNG is a preview,
not a claimed TIFF submission asset. The no-TIFF source warning is intentional.
