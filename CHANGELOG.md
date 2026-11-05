# Changelog

## Unreleased

- The Supplementary Note is built by `study/report/render.py` from `report/tables/`, `report/numbers.json` and `report/fig3_reads.json`, with its prose in `study/report/template.html`, as HTML and, new since v10, as PDF. Its text equals v10's apart from these changes: the title reads "Supplementary Note: Kinnex array integrity in FLIGHT-seq libraries" (v10: "Array integrity and TSO-TSO artifacts in Kinnex 8-fold libraries of BD Rhapsody cDNA" under a "Supplementary Note" label, which would now repeat the title and is dropped); the byline's "FIGHT-seq" reads FLIGHT-seq; section 8 and the footer describe this repository (its code folders, the package version, the commits of the run and of compute) in place of v10's report folder and its commit 0ec84dd; the code named in the captions, in the legend of Supplementary Table 2 and in Methods is this repository's (for example `study/analyses/false_cut_anatomy.py` for v10's `tools/false_cut_anatomy.py`); Methods say the inputs were extracted from the instrument archives as plain files (`study/prepare_inputs.sh`), where v10 streamed them from the archives. The number tokens these changes add or remove (the 8 of "8-fold" in the old title, and the version and commits of section 8) are listed with their reasons in `tests/golden/allowed_changes.txt`; every other number of the text equals v10's.

- `read_track_figure` drew its title from the last legend group title instead of its `title` argument (a loop variable shadowed the argument, a bug carried from v10), so every `discordant_reads.html` figure showed a stray legend title. The page's five headings now read "Reads 1 to 5 of 30: ticks are skera S-read boundaries" and so on; the Note's Fig. 3, which passes no title, is unchanged. No number changed.

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
