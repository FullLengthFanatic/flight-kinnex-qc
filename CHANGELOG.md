# Changelog

## Unreleased

- `kinnex-qc run`'s `junction_composition` row "TSO-TSO only" (summary.json `skera_pass_share`)
  differs from v10's BD144_ND number by 1 read out of 313,852 (0.0034570434472299045 vs
  0.0034602296623886418). Cause: the reader derives `skera_passing` from S-read labels alone, with
  no `--non-passing` BAM (`kinnex_qc.skera.passing_from_labels`), already measured at 20
  disagreements out of 1,969,703 ND reads against skera's real verdict (passing_from_labels
  docstring); one of those 20 reads falls in this bucket. `study/validate_reader.py` lists it in
  `KNOWN_DIFFS` so it does not fail the task 13 validation.
