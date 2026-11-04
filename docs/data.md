# Data

## Reference genome

The downstream test (`study/analyses/downstream_test.py`) maps the molecules that pass lima and isoseq refine to the GRCh38 primary assembly from GENCODE (`GRCh38.primary_assembly.genome.fa`: the GRCh38 chromosomes and scaffolds, without the alternate loci and patches), with minimap2 2.30 (`-ax splice:hq -uf --secondary=no`). The study reads the file's location from `reference_fasta` in `study/study.local.toml`; `study/study.toml` holds a placeholder.
