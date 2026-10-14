# Zhu2022 NCA raw cycling candidate

Official source: https://zenodo.org/records/6405084, CC BY4.0, version2.
Dataset1 NCA zip579629324 bytes, official MD5 cbd6d7eab2a296b52385e025e8d0f42a.
Paper DOI10.1038/s41467-022-29837-w; author's original article XML checked at
https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9046220/fullTextXML.
Table1 states66 NCA18650 cells,3.5Ah,2.65–4.2V; temperature25/35/45C;
charge0.25/0.5/1C and discharge1C for dataset1. Manufacturer/model unverified.
Treat documented original cohort separately; never automatically join NASA,
XJTU or any other NCA family merely because chemistry names resemble each other.

Download and validate entire archive before raw-feature audit. Preserve first
failed download and all range parts; do not claim training from metadata alone.
Read archive CSV streams without unsafe extraction, verify member uniqueness,
raw columns/units, time/step/current signs and exact cycle number semantics.
Infer features from early V/I only and known conditions, no Q/SOH or final
record duration. Capacity is for labels/audit only. First100 means source
cycling indices, not100 arbitrary rows or sparse diagnostic observations.

Do not copy author's max-Q2500–3500mAh filtering: it can remove legitimate
deeply aged cycles. Establish integrity/physical/protocol rules before training,
save excluded records and coverage gaps. Cells ending before origin100 should
be reported as task-ineligible, not silently discarded or evaluated at a new
origin under the original contract.

After complete coverage audit, freeze whole-cell5fold plus inner validation,
5 seeds, fixed small candidate budget. Smoke before full run, save each model,
configuration/weights/logs/predictions/metrics/curves/failures. Separate future
SOH results from original paper's current-capacity estimation task.
