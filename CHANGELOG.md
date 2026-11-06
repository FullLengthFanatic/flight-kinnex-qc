# Changelog

## Unreleased

- The Supplementary Note is built by `study/report/render.py` from `report/tables/`, `report/numbers.json` and `report/fig3_reads.json`, with its prose in `study/report/template.html`, as HTML and, new since v10, as PDF. Its text equals v10's apart from these changes: the title reads "Supplementary Note: Kinnex array integrity in FLIGHT-seq libraries" (v10: "Array integrity and TSO-TSO artifacts in Kinnex 8-fold libraries of BD Rhapsody cDNA" under a "Supplementary Note" label, which would now repeat the title and is dropped); the byline's "FIGHT-seq" reads FLIGHT-seq; section 8 and the footer describe this repository (its code folders, the package version, the commits of the run and of compute) in place of v10's report folder and its commit 0ec84dd; the code named in the captions, in the legend of Supplementary Table 2 and in Methods is this repository's (for example `study/analyses/false_cut_anatomy.py` for v10's `tools/false_cut_anatomy.py`); Methods say the inputs were extracted from the instrument archives as plain files (`study/prepare_inputs.sh`), where v10 streamed them from the archives. The number tokens these changes add or remove (the 8 of "8-fold" in the old title, and the version and commits of section 8) are listed with their reasons in `tests/golden/allowed_changes.txt`; every other number of the text equals v10's, apart from the three that the `best_cover` fix moves (next entry).

- `kinnex_qc.skera.best_cover` gave cover 1.0 to a molecule whose ZMW has no S-read, where it should give 0: the left join leaves that molecule's `qs` and `qe` null, and `min_horizontal` and `max_horizontal` skip nulls, so the overlap came out as the molecule's whole length. v10 has the same bug. `study/analyses/downstream_test.py` counts a molecule as written by skera when its best cover is at least 0.8, so on the calibration set the 6 molecules of 3 mirror reads that skera did not pass and that have no S-read (observed: ZMWs 104403695, 109185760 and 146606488) counted as written by skera. With the fix (commit "best_cover: a molecule with no S-read in its ZMW has cover 0") they count as not written. In the downstream test, "mirror reads, not written by skera" goes from 8,308 to 8,314 molecules, its lima and refine pass rates from 0.9985556 to 0.9984364 (99.9% to 99.8% in section 4 of the Note and in `docs/skera_cutting_error.md`) and its mapped rate from 0.9908522 to 0.9907385 (99.1% either way); "mirror reads, written by skera" goes from 9,575 to 9,569 molecules, its lima pass rate from 0.9994778 to 0.9995820, refine from 0.9991645 to 0.9992685 and mapped from 0.9901828 to 0.9902811 (the text prints none of these four). No molecule's lima, refine or mapping result changes, and the complete reads keep their numbers. These are the first approved changes to numbers of the Note; `tests/golden/approved_number_changes.json` lists them, and only the tests of `report/` apply it, so `tests/golden/` stays the record of v10. Only the `downstream_test` step of the full rerun was run again, with the fix, so section 8 of the Note names two run commits.

- Section 1 of the Note said "skera's next cut, at the far end of the fragment, is at a real adapter in 98%". The number (`fused_far_end_at_real_adapter`, `study/analyses/false_cut_anatomy.py`, as in v10's `tools/false_cut_anatomy.py`) measures the far end of the chimera, not of the fragment, so the sentence now reads "The chimera's other end sits at a real adapter in 98% of false cuts", as in `docs/skera_cutting_error.md`. No number changed.

- The Note's pages carry their fonts inline (base64 `@font-face`, files and licenses in `resources/fonts/`), so they load nothing over the network; v10 loaded the same families from Google Fonts. IBM Plex Sans and Mono are the weights v10 loaded. Source Serif 4 is set in the release's static optical-size cuts, Small Text for the text and Subhead for the title, where v10 used the variable font at the same optical sizes: Chrome writes a variable font into the PDF as unnamed Type 3 glyphs, the static cuts by name. The `<em>` in section 6 is now a true italic (v10 loaded no italic, so the browser slanted the roman). No number changed.

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

- New since v10: README.md and docs/ (concepts, the skera cutting error, the reader guide, data, Methods texts) and CITATION.cff. `docs/build_docs.py` writes README.md and docs/*.md from their `.md.in` sources. Every number comes from `numbers.json` through the Note's own formatter (`study/report/render.py`), except the per-library false-cut table, which reads `tables/fig1_skera_false_cut_anatomy.tsv` because numbers.json holds most of those values only as a range over the three libraries. Measurements that are not results of the study (the reader's validation runs, this machine's disk and run times) are typed in the sources; `tests/test_docs.py` lists them. The docs are built from `report/numbers.json` and `report/tables/`, those of the Note built from the full study rerun, which equal `tests/golden/v10_numbers.json` and `tests/golden/v10_tables/` apart from the commits of the run and of compute and the approved number changes listed in `tests/golden/approved_number_changes.json` (`tests/test_golden.py`). No number changed.

## Review

A per-module review of the code at commit face168. For each module group a reviewer listed correctness, edge-case and statistical findings with evidence; each finding then went to a second reviewer who tried to refute it, and the Critical and Important findings that survived were also reproduced. Line numbers are those of face168.

Groups covered (14): in `src/kinnex_qc`, scanning and annotation; classification, schemas, I/O, sampling and the worker pool; skera and the metrics; the reader's command line and runner; the reader's report, schematics and viewer. In `study/pipeline`, the config and stages 01 to 05; stage 06 and its statistics; stage 07 and the gates. In `study/analyses`, the false cuts, molecule budget, whole-cell copy and discordant reads; the mirror arms, downstream test and chain benchmark. In `study/report`, compute; render, the figures and the read tracks. Then `study/calibration` with `study/argentag`, and the study tooling (`run_study.sh`, `prepare_inputs.sh`, `compare_runs.py`, `validate_reader.py`, `make_mini.py`). Six groups had no finding: scanning, classification, the reader's runner, the reader's report, the false-cut analyses and the tooling.

35 findings: 20 confirmed, 14 refuted, 1 intended. Of the confirmed ones, 3 are fixed and 17 are not fixed. One fix changes v10 numbers; it was held for a decision and then approved.

Fixed, each with a test written first:

- `study/pipeline/config.py:110`: a skera report without the `read_segmentation.reads` attribute stopped the run with a bare StopIteration. It now raises a ValueError that names the report file and the attribute. Commit "Clear error when the skera report lacks read_segmentation.reads".
- `study/pipeline/s04_lima_refine.sh:70`: below fraction 1, a segmented BAM with no record in the hash sample stopped stage 04 with a bare StopIteration. The stage now exits with a message that names the BAM and the fraction, before lima runs. Commit "s04: clear error when the hash sample of a BAM is empty".
- `src/kinnex_qc/skera.py:107-109`, `best_cover`: a molecule whose ZMW has no S-read got cover 1.0, where it should get 0 (the left join leaves `qs` and `qe` null, and `min_horizontal` and `max_horizontal` skip nulls), so `study/analyses/downstream_test.py` counted it as written by skera. The bug comes from v10. The fix moves v10 numbers of the Note, so it was held for a decision; it was approved, and the Unreleased entry on `best_cover` lists the numbers before and after. Commit "best_cover: a molecule with no S-read in its ZMW has cover 0".
- Also fixed, from the review of the Note tracked in `report/` (not counted above): rebuilding the Note in place (`study/report/render.py --report-dir report`) writes `report/supplementary_note.print.html`, which turned `tests/test_golden.py` red, and an extra file in `report/figures/` passed that test unseen. The print page is now git-ignored, and the test checks the files git tracks in `report/`, with `figures/` holding exactly the SVG and PDF of each figure render writes. Commit "report/: ignore the print variant render writes; test the tracked files".

Not fixed (confirmed): each asks for a guard against an input the code never receives. The report compute and render steps, the figure builders, stage 07 and the ArgenTag and calibration scripts run only on the fixed BD144 and ArgenTag inputs (the study's own data, deposited), where none of these cases occurs, and the project's rules forbid error handling for cases that cannot happen. Two have a reason of their own, given on their line.

- `study/pipeline/s07_depletion.py:215`: divides by the cell's HiFi read count from the skera report with no check for 0. No BD144 cell has 0.
- `study/pipeline/s07_depletion.py:90`: in `skera_recovery`, `join_asof(strategy="nearest")` chooses by row order between two S-reads of one ZMW that start at the same base. That needs two such S-reads, and there are none (observed: in the three BD144 hash samples, 13.7, 16.5 and 16.0 million S-reads, no two S-reads of one ZMW share a start or overlap). The byte-identical rerun shows the table is stable.
- `study/pipeline/s07_depletion.py:79`: `orientation()` with no complete proper segment gives NaN and still words a prediction ("higher"). Every BD144 library has such segments.
- `study/report/compute.py:111`: divides by a library's HiFi read count with no check for 0. Every library has HiFi reads.
- `study/report/compute.py:415`: `hifi_share()` takes row 0 of a filter that could come back empty. The median lengths it looks up fall inside the length bins of the selection table.
- `study/report/compute.py:382`: the benchmark shares divide by a count that is 0 if a truth class is missing. `chain_benchmark.py` writes every class at every error rate.
- `study/report/compute.py:488`: takes row 0 of the TSO-end validation table with no check for an empty table. The table has rows.
- `study/report/compute.py:427`: divides one `hifi_share()` result by another with no check for 0 or NaN. Neither is 0 or NaN on the study's data.
- `study/report/compute.py:82`: a boolean sum is left as UInt32, against the rule to cast to Int64 before sums. The column is only averaged, never subtracted or summed further, so the overflow the rule guards against cannot happen.
- `study/report/render.py:56-57`: `sig2()` fails on 0 (log10 of 0). It formats only the TSO-TSO ratios between libraries, which are not 0.
- `study/report/figures.py:283-286`: `fig9()` divides by a library's TSO-TSO junction total with no check for 0. Every library has TSO-TSO junctions.
- `study/report/figures.py:300`: `fig9()` takes log10 of the largest flagged non-mirror share, which fails if all are 0. They are not.
- `study/report/render.py:127`: `check_claims()` divides by `partner_mean` with no check for 0. The partner means are 1,401 to 1,447 bp.
- `study/report/figures.py:42`: `load()` keeps the last of two rows with the same library and letter in the TSO-TSO letter table. The table has one row per pair.
- `study/report/render.py:283`: `note_html()` assumes `template.html` has an `<h1>`. The template is tracked and has one, and `test_title_in_one_place_of_the_template` checks it.
- `study/report/figures.py:78, 87`: `fig2()` indexes the pattern and artifact tables without `.get()`. Every library has every pattern and artifact class.
- `study/argentag/full_read_view.py:61-62`: with no flagged read, `flag_ppv` would be written empty. The ArgenTag data has flagged reads.

Rejected:

- `study/pipeline/s05_fate.py:33`, refuted: the `bc` lookup could raise KeyError on an unknown primer pair. `PAIR_NAMES` holds all four ordered pairs of the two primers lima is given.
- `study/pipeline/stats.py:162`, intended: `cutoff_summary()` needs every cutoff to be an edge, as its docstring says. `CUTOFFS` and `EDGES` are fixed constants that meet it.
- `study/analyses/downstream_test.py:95`, refuted: a molecule name is split without `[:4]`. The names come from `write_molecules`, always movie/zm/ccs/start_end.
- `study/analyses/mirror_arms.py:81`, refuted: the join would fail if the two `zm` columns had different dtypes. Both are UInt32, fixed by the parquet schemas.
- `study/report/compute.py:131-132`, refuted: `next()` without a default on the stage 06 summary. Stage 06 always writes both length kinds and all four cutoffs.
- `study/report/compute.py:138-139`, refuted: `ipw[w]` could raise KeyError. For the same reason, stage 06 always writes the 4,000 bp cutoff with both weights.
- `study/report/compute.py:107`, refuted: division by the HiFi read count. In a polars expression an integer division (`//`, the one at that line) by 0 gives null, not an error, and every library has HiFi reads.
- `study/report/compute.py:213`, refuted: division by the mirror read count. Every library has mirror reads (317,228 in ND, 1,983 in HM, 306 in kin).
- `study/report/compute.py:417`, refuted: division by `ordinary_mirrors`. It comes from `group_by().len()`, so it is at least 1.
- `study/report/compute.py:490`, refuted: division by `sreads_checked`. `tso_end_validation.py` fixes it at 200,000.
- `study/report/figures.py:193`, refuted: a key built with `{r:g}` would not match for a rate printed in scientific notation. The benchmark rates are fixed at 0.001, 0.005, 0.01 and 0.05, none printed that way.
- `study/report/figures.py:247`, refuted: the Fig. 8 bin filter could leave no row. Every library has 16 bins with more than 500 complete arrays.
- `study/calibration/build_calibration_set.py:129`, refuted: the claim that skera's S-read BAM has no `di` tag. Observed: the segmented BAM of BD144_ND (skera-split 26.1.0) carries it, and the rerun built the calibration set from it.
- `study/argentag/independent_scan.py:224-237`, refuted: the claim that it loads a 13 GB BAM into memory. It reads the ArgenTag non-passing BAM, 180,071 records.
- `study/calibration/build_calibration_set.py:61-69`, refuted: `same_zmws()` stops at the first record outside the ZMW set. That is its documented behaviour, and the calibration BAM it builds equals v10's (200,000 records, no difference).
