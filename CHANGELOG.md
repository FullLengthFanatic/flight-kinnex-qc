# Changelog

## Unreleased

- The threshold text in `gates/gate_skera.json` is now "reported only"; v10's carried an internal reference ("reported only (spec 17)"). No number changed.

- The ArgenTag `independent_scan.txt` keeps its aggregate lines only. v10's listed, under "Disagreement details (first 10):", the 8 reads where the independent scan and the study disagree, by ZMW id; those ids belong to a file ArgenTag shared privately, so the list now goes to `independent_scan_disagreements.tsv` in the ArgenTag folder, never next to the published text. The heading line stays, with no rows under it. No number changed.

- `study/calibration/build_calibration_set.py` writes `nd_first200k.hifi.nomod.bam` straight from BD144_ND's HiFi BAM. v10's copy was cut with samtools 1.13 through SAM text (its @PG lines say so), which keeps 6 significant digits of the float tags `ec`, `rq` and `sn` and changes the type codes of integer tags. So the new file differs from v10's in the last digits of those float tags and in those type codes, and `rq` in `nd_first200k_truth/read_summary.parquet` differs in 187,115 of the 200,000 reads; names, sequences, qualities, the other tag values and the other three truth tables are the same. Sent through the same samtools trip, the new records equal v10's exactly. The report's calibration numbers were reproduced from v10's calibration files.

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
