#!/usr/bin/env python3
"""Where the 8 slots of a Kinnex 8-fold HiFi read go, per HiFi read, for one library.

Per HiFi read: slots the array does not fill (8 minus the molecules our segmentation finds), artifact molecules
(TSO-TSO, R1-R1, one primer or none, fused), and usable molecules (proper, not fused). Each molecule is matched to
skera's S-reads by their parent-read coordinates (skera.budget_classes, the rule of mirror_arms.py): a usable molecule is
written CLEAN when one S-read covers at least 80% of it and at least 20% of no other molecule. A clean S-read's fate
comes from stage 05 (lima_pass, refine_asrun, mapped_fltnc). So in the standard workflow (skera, lima, refine, mapping)
a usable molecule is kept (its clean S-read is in the mapped fltnc BAM), lost after skera (clean S-read not in the
mapped BAM; split by the first step it fails in our lima and refine run, else mapping) or never written cleanly (no clean
S-read: not written, fused with a neighbour or partly covered).
Also counts mapped S-reads that are not the clean S-read of a usable molecule (split, fused or artifact S-reads), so the
budget can be reconciled with the mapped fltnc count.
Writes <label>_budget.json (per HiFi read averages) and <label>_budget_by_pattern.tsv (the same by adapter-chain pattern).
Usage: study/analyses/molecule_budget.py --sample-dir <pipeline sample dir> --label <name> --out <dir>
"""
import argparse
import json
from pathlib import Path

import polars as pl

from kinnex_qc.skera import budget_classes, budget_per_read

SLOTS = 8
COLS = ["molecules", "unfilled_slots", "artifact_tso_tso", "artifact_r1_r1", "artifact_one_primer_or_none", "artifact_fused",
        "usable", "kept_standard", "lost_lima", "lost_refine", "lost_mapping", "not_written_cleanly", "other_mapped"]


def budget_reads(rs: pl.DataFrame, segs: pl.DataFrame, sk: pl.DataFrame) -> pl.DataFrame:
    """Per HiFi read of rs (zm, primary_pattern): the counts of COLS. segs: the reads' molecules (zm, start, end,
    end_class, internal); sk: their S-reads (zm, di, qs, qe) with fates (lima_pass, refine_asrun, mapped_fltnc, no nulls)."""
    cls = budget_classes(segs, sk.select("zm", "di", "qs", "qe"))
    m = cls.join(sk.select("zm", pl.col("di").alias("clean_di"), "lima_pass", "refine_asrun", "mapped_fltnc"),
                 on=["zm", "clean_di"], how="left")
    written = pl.col("usable") & pl.col("clean_di").is_not_null()
    used_sreads = m.filter(written).select("zm", pl.col("clean_di").alias("di")).unique()
    other_mapped = (sk.filter(pl.col("mapped_fltnc")).join(used_sreads, on=["zm", "di"], how="anti")
                    .group_by("zm").len().rename({"len": "other_mapped"}))
    lost = written & ~pl.col("mapped_fltnc").fill_null(False)   # clean S-read, not in the mapped BAM
    fates = m.group_by("zm").agg(
        (pl.col("usable") & pl.col("mapped_fltnc").fill_null(False)).sum().alias("kept_standard"),
        (lost & ~pl.col("lima_pass").fill_null(False)).sum().alias("lost_lima"),
        (lost & pl.col("lima_pass").fill_null(False) & ~pl.col("refine_asrun").fill_null(False)).sum().alias("lost_refine"),
        (lost & pl.col("lima_pass").fill_null(False) & pl.col("refine_asrun").fill_null(False)).sum().alias("lost_mapping"))
    per_read = budget_per_read(cls).rename({"usable_not_clean": "not_written_cleanly"}).drop("usable_clean").join(fates, on="zm")
    return (rs.join(per_read, on="zm", how="left").join(other_mapped, on="zm", how="left")
            .with_columns(pl.exclude("zm", "primary_pattern").fill_null(0).cast(pl.Int64))
            .with_columns((SLOTS - pl.col("molecules")).alias("unfilled_slots")))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", required=True, help="pipeline output directory of one sample")
    ap.add_argument("--label", required=True, help="name for the output files")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)
    o, out = Path(args.sample_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rs = (pl.read_parquet(o / "read_summary.parquet", columns=["zm", "source", "primary_pattern"])
          .filter(pl.col("source") == "hifi").drop("source"))
    segs = (pl.read_parquet(o / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class", "internal"])
            .filter(pl.col("source") == "hifi").drop("source"))
    sk = (pl.read_parquet(o / "segments.parquet", columns=["zm", "di", "qs", "qe"])
          .join(pl.read_parquet(o / "fate.parquet", columns=["zm", "di", "lima_pass", "refine_asrun", "mapped_fltnc"]), on=["zm", "di"], how="left")
          .with_columns(pl.col("lima_pass", "refine_asrun", "mapped_fltnc").fill_null(False)))
    reads = budget_reads(rs, segs, sk)
    summary = {"label": args.label, "hifi_reads": reads.height, **{c: float(reads[c].mean()) for c in COLS},
               "mapped_sreads": float(sk.filter(pl.col("mapped_fltnc")).height / reads.height)}
    (out / f"{args.label}_budget.json").write_text(json.dumps(summary, indent=1) + "\n")
    (reads.group_by("primary_pattern").agg(pl.len().alias("reads"), *[pl.col(c).mean() for c in COLS]).sort("primary_pattern")
     .write_csv(out / f"{args.label}_budget_by_pattern.tsv", separator="\t"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
