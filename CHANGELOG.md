# Changelog

## Unreleased

- `discordant_reads.html` carries plotly inline (about 4.7 MB), so the review page opens without a network; v10's page loaded plotly from the plotly CDN. Its six figures hold the same data and layout as v10's (the same 30 reads in the same order). No number changed.

- The threshold text in `gates/gate_skera.json` is now "reported only"; v10's carried an internal reference ("reported only (spec 17)"). No number changed.

- The ArgenTag `independent_scan.txt` holds aggregate lines only. v10's listed, under "Disagreement details (first 10):", the 8 reads where the independent scan and the study disagree, by ZMW id; those ids belong to a file ArgenTag shared privately, so the list now goes to `independent_scan_disagreements.tsv` in the run's private ArgenTag output folder, next to the other non-public per-read tables (`argentag_sread_coords.parquet`, `full_reads/`). The heading line stays, with no rows under it. No number changed.

- The row order of `q2/tables/lima_vs_class.tsv` is now fixed for rows tied on `end_class` and count (`lima_pair` is a third sort key), so the table is byte-stable across reruns. No value changed.

- The `confusion` rows of `gates/gate_lima.json` tied on count are now ordered by `lima_pair`, then `end_class`, so the file is byte-stable across reruns. This is the order v10's files have (one tie, in BD144_kin). No value changed.

- `kinnex-qc run`'s `junction_composition` row "TSO-TSO only" (summary.json `skera_pass_share`)
  differs from v10's BD144_ND number by 1 read out of 313,852: the reader reports
  0.0034570434472299045 against v10's 0.0034602296623886418. Cause: the reader derives
  `skera_passing` from the S-read adapter labels alone (`kinnex_qc.skera.passing_from_labels`),
  because it reads only the segmented BAM, while v10 took skera's verdict from the non-passing
  BAM; the two disagree on 20 of BD144_ND's 1,969,703 sampled reads (19 truncated, 1 mirror, none
  of the complete arrays), and that one mirror read falls in this bucket. Reviewed and accepted:
  `study/validate_reader.py` pins both values and prints the line as KNOWN, and
  `tests/test_reader_validation.py` (real) passes.
