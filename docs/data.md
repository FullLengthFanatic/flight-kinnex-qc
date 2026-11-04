# Data

## Reference genome

The downstream test (`study/analyses/downstream_test.py`) maps the molecules that pass lima and isoseq refine to the GRCh38 primary assembly from GENCODE (`GRCh38.primary_assembly.genome.fa`: the GRCh38 chromosomes and scaffolds, without the alternate loci and patches), with minimap2 2.30 (`-ax splice:hq -uf --secondary=no`). The study reads the file's location from `reference_fasta` in `study/study.local.toml`; `study/study.toml` holds a placeholder. The analysis takes it from `--ref`, which defaults to `reference_fasta` and stops with a one-line error when neither is set.

## ArgenTag MDA-MB-453 8-fold

The ArgenTag scripts (`study/argentag/`) read one folder, `argentag_dir` in `study/study.local.toml` (`study/study.toml` holds a placeholder), with two BAMs in `data/`:

- `segmented.bam`: the public S-reads of the 8-fold cell (skera 1.2.0, as the BAM's @PG line says), from https://downloads.pacbcloud.com/public/dataset/Kinnex-single-cell-RNA/DATA-RevioSPRQ-Kinnex-ArgenTag-MDAcellLine/MDA-8fold/0-Sreads/
- `segmented.non_passing.bam`: skera's file of the reads it did not pass for the same cell. It is not public; ArgenTag shared it with us on 2026-10-05, and it is not redistributed here.

The scripts write into `analysis/` of that folder. The cell's HiFi read count (1,964,147) comes from Table 1 of the ArgenTag/PacBio technical note.

## Calibration set

The calibration scripts (`study/calibration/`) write into `calibration_dir` in `study/study.local.toml`. `build_calibration_set.py` takes the first 200,000 HiFi reads of BD144_ND from the BAMs of `study/samples.tsv`; `skera_split.py` runs skera 1.2.0 (bioconda pbskera=1.2.0, `envs/skera120.yml`) on them, in the conda environment named by `skera120_env`.
