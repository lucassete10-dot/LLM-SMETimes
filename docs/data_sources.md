# Battery aging data provenance and intake

This is a source/access audit, not a claim that new data have entered training.
The currently trained v2 models still use only NASA (4 cells), CALCE (4),
MIT/MATR (134), and XJTU (55). Dataset-level label and censoring rules are in
[`battery_data_v2/README.md`](../battery_data_v2/README.md); the authoritative
cell counts are in `battery_data_v2/manifest.json`.

| Candidate | Why useful | Original source / rights | Intake status |
| --- | --- | --- | --- |
| Panasonic NCR18650B, 116 cells, 37 profiles | Cross-condition transfer; repeated reference tests | [Zenodo original record](https://zenodo.org/records/7896429), CC BY 4.0 | Seven cells downloaded and MD5-verified; test-level parser and QA added. Three L214 cells have interval-observed 80% events; not in training. |
| HUST, 77 LFP cells | Within-chemistry protocol transfer | [Mendeley original record](https://data.mendeley.com/datasets/nsc7hnsg4s/2), CC BY 4.0 | Public page exists; unauthenticated file API requested sign-in. Manual permitted download needed. |
| RWTH, 48 NMC cells | Replicate-cell generalization, raw V/I/T | [RWTH original record](https://publications.rwth-aachen.de/record/818642/), CC BY 4.0 | Official ZIP returned a site challenge to this client. Do not bypass it. |
| Oxford Degradation Dataset 1, 8 pouch cells | Independent long-life dataset | [Oxford Research Archive](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) | Archive returned HTTP 403; rights and raw schema still need direct verification. |
| Large-format LFP cyclic/calendar, 12 cells | Temperature and calendar aging, 180 Ah scale | [Zenodo original record](https://zenodo.org/records/22939281), CC BY 4.0 | Candidate only; 2.5 GB archive not yet downloaded. Both paper and Zenodo citation required. |
| Bayreuth degradation indicators, 48 cells | EIS, pulses, OCV, capacity health indicators | [Zenodo original record](https://zenodo.org/records/15755725), CC BY 4.0 | Candidate only; 70-cycle aging is too short to assume observed 80% EOL. |
| NASA randomized battery usage | Dynamic/random-walk load profiles | [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | Distinct from current NASA four-cell set; license, raw schema, and capacity labels not yet verified. |

The full machine-readable candidate fields and explicit unknowns are in
[`metadata/dataset_registry.yaml`](../metadata/dataset_registry.yaml), with
rights tracked separately in [`metadata/licenses.yaml`](../metadata/licenses.yaml).
`null` means not yet verified. A record must **not** enter training until its
raw files, checksum, parser, SOH reference, endpoint/censoring rule, QA, and
whole-cell identity have been checked.

## Duplication and leakage gate

- Use original dataset DOI plus source file checksum as the source identity.
  A GitHub/Kaggle mirror of NASA, CALCE, XJTU, or MIT is not a new domain.
- Use dataset ID + original cell ID for each cell. Resolve any continuation
  files before assigning splits; this is already relevant to MIT batches.
- Keep all cycles of each cell together, and avoid source mirror/derived
  feature copies crossing train/test. Do not infer EOL where the source has
  not observed an 80% crossing.
- The [Panasonic experimental description](https://zenodo.org/records/7896429/files/Galijatovic_2023_Ageing%20Data%20of%20Lithium-ion%20Panasonic%20NCR18650B%20Battery%20Cells.pdf)
  says the cells are NCA, nominal 3.25 Ah, and the 25°C RTP repeats a 1C
  discharge to 2.5 V twice. Its original EOL was 70% capacity **or project
  end**, not our unified 80% target. Do not convert a calendar-aging cell to
  "remaining cycles" or treat a threshold crossing between sparse RTPs as an
  exact event cycle.

## Next intake steps

1. Extend Panasonic beyond one high-stress profile. L214 cells `2_101/102/103`
   provide three interval-observed 80% events at 310–355 equivalent full
   cycles (EFC), not exact cycle events. `2_108` remains above 80% SOH;
   `2_119` has only one RTP25. Preserve these censoring distinctions.
2. Map protocol-qualified RPT capacity candidates and raw waveform phases into
   canonical cell/diagnostic rows. The current EFC integral excludes RTP
   throughput and keeps interval-censored EOL; audit cross-cell reproducibility
   and cycle segmentation before deriving any point RUL label.
3. Check overlap with existing data by source and signal fingerprints, then
   hold out entire cells/profiles for evaluation.
