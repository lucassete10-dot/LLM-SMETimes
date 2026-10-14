# Battery data coverage (ingested v2 only)

Generated from `battery_data_v2/manifest.json`,
`metadata/dataset_registry.yaml`, and the separately named endpoint audit
when available. Candidate data are **not** counted as
training cells. An observation is a retained capacity diagnostic, not
necessarily a full cycling event.

| Dataset | Chemistry (cells) | Cells | Capacity observations | SOH labels | Source-contract endpoints (not unified 80% EOL) |
| --- | --- | ---: | ---: | ---: | ---: |
| CALCE | LiCoO2 cathode: 4 | 4 | 3778 | 3774 | 4 |
| MIT | LFP/graphite: 134 | 134 | 109878 | 109744 | 122 |
| NASA | unknown: 4 | 4 | 636 | 632 | 4 |
| XJTU | NCM532: 55 | 55 | 10389 | 10114 | 55 |

Total: 197 cells, 124681 retained capacity observations.

The source-contract endpoint field follows each source's original
label convention; it does **not** count confirmed 80%-SOH threshold
events. In particular, the 55 XJTU entries must not be interpreted
as 55 observed project-defined RUL-to-80% labels.

## Confirmed per-cell 80%-SOH endpoints (audited separately)

| Dataset | Confirmed 80% events | Right-censored for 80% | Source-defined endpoints |
| --- | ---: | ---: | ---: |
| CALCE | 4 | 0 | 4 |
| MIT | 43 | 91 | 122 |
| NASA | 4 | 0 | 4 |
| XJTU | 0 | 55 | 55 |

These are observed events under the audited three-consecutive-
diagnostic confirmation rule, not necessarily precise physical
crossing times. A censored cell has no known 80% RUL point target.

## Condition and waveform coverage by dataset

| Dataset | Ambient temperature (cells) | Charge C-rate (cells) | Discharge C-rate (cells) | DoD | Measured temperature waveform (cells) |
| --- | --- | --- | --- | --- | ---: |
| CALCE | unknown: 4 | 0.5: 4 | 1.0: 4 | not structured in v2 | 0/4 |
| MIT | 30: 134 | unknown: 134 | 4.0: 134 | not structured in v2 | 134/134 |
| NASA | unknown: 4 | unknown: 4 | unknown: 4 | not structured in v2 | 4/4 |
| XJTU | unknown: 55 | 2.0: 16, 3.0: 15, unknown: 24 | 1.0: 23, unknown: 32 | not structured in v2 | 55/55 |

V/I waveforms are available through the existing per-dataset raw adapters,
but this report does not assert their sampling completeness. CALCE's
accepted Step 7 diagnostics do not contain measured temperature. The
missing chemistry for NASA is intentionally not guessed. The v2 manifest
does not structure per-cell DoD or MIT multi-stage charge rate into one
scalar C-rate; `unknown` is not evidence of a zero rate.

## Candidate domains, not yet ingested

| Candidate | Chemistry | Claimed cells | Temperature conditions | Raw V/I/T claimed | Access status |
| --- | --- | ---: | --- | --- | --- |
| PANASONIC_NCR18650B | NCA | 116 | 0, 5, 20, 30, 40, and 45 degC aging profiles; RTP at 5, 15, 25, 35, 45 degC initially | yes/yes/yes | sample_downloaded_test_level_qa_not_training_ingested |
| HUST_77 | LFP/graphite | 77 | 30 degC | ?/?/? | manual_download_needed_not_ingested |
| RWTH_UR18650E | NMC/graphite | 48 | same controlled conditions; numerical value pending verification | yes/yes/yes | automated_download_challenged_not_ingested |
| OXFORD_DEGRADATION_1 | unknown | 8 | unknown | ?/?/? | official_page_unavailable_not_ingested |
| ZENODO_LFP_STATIONARY_2026 | LFP | 12 | 35 and 50 degC | yes/yes/? | candidate_not_downloaded |
| BAYREUTH_PATH_2025 | unknown | 48 | varied; details in accompanying metrics workbook, not yet verified | ?/?/? | candidate_not_downloaded |
| NASA_RANDOM_USAGE | unknown | unknown | room-temperature and 40 degC random-walk variants listed by NASA | ?/?/? | rights_and_schema_review_needed_not_downloaded |

## Panasonic raw intake sample (excluded from training)

| Cell | Aging profile | Aging mode | Qualified 25°C RPTs | 80% crossing evidence | Invalid temperature points |
| --- | --- | --- | ---: | --- | ---: |
| 2_018 | C06 | near_calendar | 21 | between tests 580 and 582; not exact; calendar, no cycle-RUL | 0 |
| 2_097 | L206 | calendar | 16 | between tests 4821 and 4823; not exact; calendar, no cycle-RUL | 45 |
| 2_101 | L214 | cyclic | 22 | between tests 5050 and 5052; not exact; EFC 334.1–354.4 | 48 |
| 2_102 | L214 | cyclic | 22 | between tests 5103 and 5106; not exact; EFC 310.7–332.5 | 48 |
| 2_103 | L214 | cyclic | 22 | between tests 5159 and 5162; not exact; EFC 310.2–332.1 | 47 |
| 2_108 | L207 | cyclic | 23 | not observed/confirmed | 51 |
| 2_119 | L220 | cyclic | 1 | not observed/confirmed | 0 |

RPT count is not cycling count. Provisional capacities are the median of
two protocol-qualified 1C discharges; calendar-aging cells cannot be
assigned cycle-RUL. Source EOL was 70% or project end, whereas the
project target is a separately audited 80% crossing.

## Gaps that matter for the model

- No additional source has passed raw-file QA, label harmonization, or
  cell-level deduplication yet; cross-dataset claims remain four-source.
- Ambient temperature and DoD are largely unstructured in the current
  manifest. The dataset cannot yet support a fair temperature/DoD ablation.
- The four sources use different observed endpoint rules; raw RUL cycle
  errors must not be pooled until a common 80%-SOH EOL task is audited.
- Panasonic would add within-format condition diversity; HUST would add
  LFP discharge protocols; RWTH would add NMC replicate-cell diversity;
  the large-format LFP source would add calendar-aging stress.
