# Data

## Reference genome

The downstream test (`study/analyses/downstream_test.py`) maps the molecules that pass lima and isoseq refine to the GRCh38 primary assembly from GENCODE (`GRCh38.primary_assembly.genome.fa`: the GRCh38 chromosomes and scaffolds, without the alternate loci and patches), with minimap2 2.30 (`-ax splice:hq -uf --secondary=no`). The study reads the file's location from `reference_fasta` in `study/study.local.toml`; `study/study.toml` holds a placeholder. The analysis takes it from `--ref`, which defaults to `reference_fasta` and stops with a one-line error when neither is set.

## ArgenTag MDA-MB-453 8-fold

The ArgenTag scripts (`study/argentag/`) read one folder, `argentag_dir` in `study/study.local.toml` (`study/study.toml` holds a placeholder), with two BAMs in `data/`:

- `segmented.bam`: the public S-reads of the 8-fold cell (skera 1.2.0, as the BAM's @PG line says), from https://downloads.pacbcloud.com/public/dataset/Kinnex-single-cell-RNA/DATA-RevioSPRQ-Kinnex-ArgenTag-MDAcellLine/MDA-8fold/0-Sreads/
- `segmented.non_passing.bam`: skera's file of the reads it did not pass for the same cell. It is not public; ArgenTag shared it with us on 2026-10-05, and it is not redistributed here.

The scripts only read that folder. They write into the folder given by `--out`, which has no default: `sread_coords.py` and `annotate_non_passing.py` write the per-read tables (`argentag_sread_coords.parquet`, `full_reads/`), `full_read_view.py` and `independent_scan.py` read them from there and add their aggregate tables and `independent_scan.txt`, and `independent_scan.py` adds its per-read disagreement list (`independent_scan_disagreements.tsv`). The per-read tables hold ArgenTag read ids and are never published. The cell's HiFi read count (1,964,147) comes from Table 1 of the ArgenTag/PacBio technical note.

## Calibration set

The calibration scripts (`study/calibration/`) write into the folder given by `--out`, which has no default; they never write into `calibration_dir` in `study/study.local.toml`, which holds v10's calibration set, the reference their outputs are compared with. `build_calibration_set.py` takes the first 200,000 HiFi reads of BD144_ND from the BAMs of `study/samples.tsv` through samtools 1.13 with htslib 1.13, as v10 did (its docstring says why the versions matter): `samtools` in `study/study.local.toml` names the executable, else it runs in the conda environment named by `samtools_env` (`envs/samtools113.yml`). `skera_split.py` runs skera 1.2.0 (bioconda pbskera=1.2.0, `envs/skera120.yml`) on them, in the conda environment named by `skera120_env`; it, `head_version_check.py` and `tso_end_validation.py` read the set from `--calibration-dir`, which defaults to `calibration_dir`.
