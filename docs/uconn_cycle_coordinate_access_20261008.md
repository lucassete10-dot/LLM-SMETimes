# UConn cycle-coordinate follow-up

Public primary dataset records checked again:

- https://digitalcommons.lib.uconn.edu/reil_datasets/1/
- https://digitalcommons.lib.uconn.edu/reil_datasets/5/

Both pages list64 cells and RPT approximately every3.5days (~100cycles depending on condition). This is not evidence that each RPT is exactly100cycles. The two records link to the same README attachment endpoint article1000/filename0 and protocol CSV filename2. Current retrieval attempts returned timeout/403; no content obtained. No access-control bypass attempted.

Author source `postprocessing.py` maps `num_cycles` to the maximum of `Num Cycles` in each RPT. Local processed data contain decreasing values across RPTs. Thus direct use as a cumulative cycle coordinate is unverified. Blind summation, using100*RPT or sorting by decreasing cycle values would fabricate chronology and is not permitted.

Need verified README/raw cycling semantics or author evidence for the cumulative coordinate and protocol mapping. The existing processed pulse arrays are not yet approved for source-cycle-based future prediction. RPT-index experiments, if used as a supplement, must be separately named and must not be presented as the requested N-source-cycle result. Other four cohorts can progress independently.
