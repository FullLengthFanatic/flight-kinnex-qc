# kinnex-qc reader guide

This section covers what `kinnex-qc run` checks, how to install and run it, and how it has been
validated.

## What `kinnex-qc run` checks

`kinnex-qc run` reads a HiFi (CCS) BAM from a Kinnex 8-fold array and classifies the reads of a
hash sample (`--fraction`, default 10% of ZMWs) by their MAS adapter chain: whether each is a
complete array, a mirror array, truncated, or something else, and what share of its molecules are
proper cDNA versus an artifact (a TSO-TSO or R1-R1 junction, a molecule missing a primer, or an
internal fusion). The hash is salted the same way the study's own analysis sampled reads, so a
rerun on the same file picks the same sample. Given skera's S-read BAM too (`--sreads`), it also
checks skera's own cutting on that same sample: where a cut lands away from the adapter it should
sit on, counted as a false cut (`kinnex_qc.skera.false_cuts`), and how many usable molecules a
read's slot budget yields after artifacts and cuts that were not written cleanly
(`kinnex_qc.skera.budget_stats`). With `--write-molecules`, it writes molecules for every read in
the file, not only the sampled ones.

The command writes four files into `--out`: `summary.json` (every number above), `reads.tsv.gz`
(one row per sampled read), `report.html` (a short visual summary with example reads) and
`run.log`.

## Install and run

Build the wheel and install it:

```
uv build --wheel
pip install dist/kinnex_qc-0.1.0-py3-none-any.whl
```

Then run it against a HiFi BAM and, if you have one, skera's S-read BAM:

```
kinnex-qc run --hifi hifi_reads.bam --sreads segmented.bam --fraction 0.1 --threads 8 --out OUT_DIR
```

`--fraction` (default 0.1) is the share of reads the summary and report describe; `--threads`
(default 4) is the number of worker processes. `--sreads` is optional: without it, `summary.json`
has no `skera` section. See `kinnex-qc run --help` for the rest of the options.

## Validation

`kinnex-qc run` is validated on BD Rhapsody (FLIGHT-seq) Kinnex 8-fold libraries only. No 10x
Kinnex HiFi dataset has been checked against it yet.

The check: `study/validate_reader.py` ran commit `1a33bbf` (package version 0.1.0) against three
BD Rhapsody libraries, BD144_ND, BD144_HM and BD144_kin, at `--fraction 0.18` and `--threads 10`,
and compared 229 numbers from each run's `summary.json` (129 for BD144_ND, 50 each for BD144_HM
and BD144_kin) against the frozen tables from the same study's own, independent analysis
(`tests/golden`): patterns, artifact rates, false cut letters and the molecule budget in all three
libraries, plus BD144_ND's mirror junction letters, junction composition, array size rule and
mirror arm tables. BD144_HM and BD144_kin have no v10 tables for those last four, so they are not
compared; `skera.complete_reads`, `skera.budget.artifact` and `skera.budget.usable_clean` also
have no v10 counterpart and are not compared in any library. Later commits changed how
`--write-molecules` streams its output and stopped defaulting `--fraction` to 1 when it is set
without an explicit `--fraction`, added progress lines to `run.log`, changed row order on ties,
and changed how `mirror_size_rule` is computed (it now goes through
`skera.single_tso_tso_mirrors`); none of them touch `summary.json`'s numbers, which their own
tests confirm against a fixed calibration BAM.

Across the three libraries, 33,628,976 HiFi reads were read from the BAMs and 6,054,028 of them
fell in the 0.18 hash sample the summary and report describe. One of the 229 compared numbers
differs from v10: BD144_ND's mirror junction composition "TSO-TSO only" bucket's
`skera_pass_share` is 0.0034570434472299045 against v10's 0.0034602296623886418, one read out of
313,852. The cause is that the reader has no `--non-passing` option, so it derives skera's pass or
fail verdict from the S-read labels alone (`kinnex_qc.skera.passing_from_labels`), which disagrees
with skera's own verdict on 20 of BD144_ND's 1,969,703 sampled reads (19 truncated, 1 mirror, none
of the complete arrays); that one mirror read falls in this bucket. This difference was reviewed
and accepted: `study/validate_reader.py` pins both values and prints the line as KNOWN, not a
failure, and `tests/test_reader_validation.py` passes. It is the only one of the 229 compared
numbers that differs; every other number matches v10 exactly. See `CHANGELOG.md` for the entry.

Run time and peak memory, as measured on the machine that ran these three checks (a machine shared
with other jobs at the time, load 37 to 57 on 32 CPUs), at `--fraction 0.18` and `--threads 10`,
from `/usr/bin/time -v`: this is what these three runs cost on that machine, not a benchmark.

| library | wall time | peak memory |
| --- | --- | --- |
| BD144_ND | 3:45:04 | 14,797,320 kB (14.1 GiB) |
| BD144_HM | 4:11:28 | 17,228,784 kB (16.4 GiB) |
| BD144_kin | 4:11:08 | 16,728,116 kB (16.0 GiB) |
