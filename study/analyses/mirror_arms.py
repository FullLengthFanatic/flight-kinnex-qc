#!/usr/bin/env python3
"""What skera writes from mirror reads with a single TSO-TSO junction, molecule by molecule.

A mirror read Q' .. X' [TSO-TSO] X .. Q carries reverse-complement adapters on the arm BEFORE the junction molecule and
forward adapters on the arm AFTER it (skera.single_tso_tso_mirrors). Molecules come from our full-read annotation
(annot_segments of a truth or sample directory); S-reads from skera's own tags (zm, di, qs, qe, dl, dr), given as a
segments parquet or as a skera S-read BAM (read as the reader command reads one, di the record's index within its ZMW;
observed in the skera 1.2.0 S-read BAM of the calibration set too: the S-reads of one ZMW are consecutive records).
Per proper, unfused molecule, skera.mirror_arm_classes and skera.molecule_coverage: CLEAN if one S-read covers at least
80% of it and at least 20% of no other molecule; FUSED if an S-read covers at least 80% of it and at least 20% of
another molecule; PARTIAL if the best S-read covers 20 to 80%; NOT_WRITTEN otherwise. Per read: NO_SREAD;
SPANS_JUNCTION if an S-read covers at least 20% of the TSO-TSO molecule; otherwise AFTER_ONLY, BEFORE_ONLY or BOTH_ARMS
by the arms whose molecules S-reads cover at least 20% of; OTHER if the S-reads cover no molecule that much. Complete
arrays (adapters A to Q in order) are classified the same way as a baseline.
Optional: skera's non_passing BAM (otherwise skera_passing from read_summary), its found_adapters.csv.gz (which of our
adapters skera found, matched within 10 bp, by position relative to the junction) and its ligations.csv.
Writes <label>_molecules.tsv, <label>_reads.tsv, <label>_reads_by_letter.tsv, <label>_orientation.tsv,
<label>_summary.json and, with --found-adapters, <label>_found_adapters.tsv into --out.
Usage: study/analyses/mirror_arms.py --truth <dir> (--segments <parquet> | --sreads <S-read BAM>) --label <name> --out <dir>
       [--non-passing <BAM>] [--found-adapters <csv.gz>] [--ligations <csv>] [--baseline-fraction 1.0]
"""
import argparse
import gzip
import json
from pathlib import Path

import polars as pl

from kinnex_qc.io import iter_bam
from kinnex_qc.molecules import USABLE
from kinnex_qc.runner import read_sreads
from kinnex_qc.sampling import in_hash_sample
from kinnex_qc.skera import COVER, TOUCH, mirror_arm_classes, molecule_coverage, single_tso_tso_mirrors, sread_overlaps

JUNCTION_TOL = 5        # bp: an adapter of ours ending (starting) this close to the junction molecule's start (end) flanks it
FOUND_TOL = 10          # bp: a found_adapters entry matches an adapter of ours when either end is this close


def shares(df: pl.DataFrame, by: list[str], key: str) -> pl.DataFrame:
    return (df.group_by(*by, key).len().rename({"len": "n"})
            .with_columns((pl.col("n") / pl.col("n").sum().over(by)).alias("share")).sort(*by, key))


def read_classes(segs: pl.DataFrame, st: pl.DataFrame, sk: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """For the single TSO-TSO mirror reads st: one row per read with its letter and read_class, and the S-reads
    covering at least COVER of a molecule, by the molecule's region (junction, before, after, other) and the S-read's
    orientation (orientation table)."""
    mols = (segs.join(st, on="zm").with_columns(
        pl.when((pl.col("start") == pl.col("js")) & (pl.col("end") == pl.col("je"))).then(pl.lit("junction"))
        .when(pl.col("end") <= pl.col("js")).then(pl.lit("before")).when(pl.col("start") >= pl.col("je")).then(pl.lit("after"))
        .otherwise(pl.lit("other")).alias("region")).select("zm", "start", "end", "region"))
    msk = sk.filter(pl.col("zm").is_in(st["zm"]))
    pairs = sread_overlaps(mols, msk)
    arms = (pairs.filter(pl.col("ov") >= TOUCH).group_by("zm")
            .agg((pl.col("region") == "junction").any().alias("j"), (pl.col("region") == "before").any().alias("b"),
                 (pl.col("region") == "after").any().alias("a")))
    reads = (st.select("zm", "letter").join(arms, on="zm", how="left")
             .join(msk.group_by("zm").len().rename({"len": "n_sreads"}), on="zm", how="left")
             .with_columns(pl.when(pl.col("n_sreads").is_null()).then(pl.lit("NO_SREAD")).when(pl.col("j")).then(pl.lit("SPANS_JUNCTION"))
                           .when(pl.col("b") & pl.col("a")).then(pl.lit("BOTH_ARMS")).when(pl.col("a")).then(pl.lit("AFTER_ONLY"))
                           .when(pl.col("b")).then(pl.lit("BEFORE_ONLY")).otherwise(pl.lit("OTHER")).alias("read_class")))
    orientation = (pairs.filter(pl.col("ov") >= COVER).with_columns((pl.col("dl") < pl.col("dr")).alias("ascending"))
                   .group_by("region", "ascending").agg(pl.struct("zm", "di").n_unique().alias("sreads")).sort("region", "ascending"))
    return reads, orientation


def found_adapters(features: pl.DataFrame, st: pl.DataFrame, path: str) -> tuple[pl.DataFrame, dict]:
    """Which of our adapters in the mirror reads st skera's found_adapters.csv.gz lists, by where they sit relative to the
    junction; and the number of its entries for these reads, and of those that match none of our adapters."""
    feats = (features.filter(pl.col("kind") == "MAS").join(st, on="zm").with_columns(
        pl.when((pl.col("end") - pl.col("js")).abs() <= JUNCTION_TOL).then(pl.lit("junction_reverse"))
        .when((pl.col("start") - pl.col("je")).abs() <= JUNCTION_TOL).then(pl.lit("junction_forward"))
        .when(pl.col("end") < pl.col("js")).then(pl.lit("before_arm")).when(pl.col("start") > pl.col("je")).then(pl.lit("after_arm"))
        .otherwise(pl.lit("inside_junction_molecule")).alias("where")).with_row_index("hid"))
    want, rows = set(st["zm"].to_list()), []
    with gzip.open(path, "rt") as fh:
        next(fh)
        for line in fh:
            name, _, ai, s, e = line.rstrip("\n").split(",")
            zm = int(name.split("/")[1])
            if zm in want:
                rows.append((zm, int(ai), int(s), int(e)))
    found = pl.DataFrame(rows, schema=["zm", "adapter_index", "fs", "fe"], orient="row").with_columns(pl.col("zm").cast(pl.UInt32)).with_row_index("fid")
    match = (feats.select("hid", "zm", "start", "end").join(found, on="zm")
             .filter(((pl.col("fs") - pl.col("start")).abs() <= FOUND_TOL) | ((pl.col("fe") - pl.col("end")).abs() <= FOUND_TOL)))
    table = (feats.with_columns(pl.col("hid").is_in(match["hid"].unique()).alias("found"))
             .group_by("where").agg(pl.len().alias("our_adapters"), pl.col("found").mean().alias("share_found_by_skera")).sort("where"))
    return table, {"skera_found_entries": found.height, "skera_found_at_none_of_our_adapters": found.height - match["fid"].n_unique()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="directory with read_summary, read_features and annot_segments parquets")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--segments", help="skera's S-reads as a segments parquet")
    src.add_argument("--sreads", help="skera's S-reads as an S-read BAM")
    ap.add_argument("--label", required=True, help="name for the output files")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--non-passing", help="skera's non-passing BAM of the same reads")
    ap.add_argument("--found-adapters", help="skera's found_adapters.csv.gz")
    ap.add_argument("--ligations", help="skera's ligations.csv")
    ap.add_argument("--baseline-fraction", type=float, default=1.0,
                    help="complete arrays used as baseline: ZMWs in the salted hash sample of this fraction")
    args = ap.parse_args(argv)
    t, out = Path(args.truth), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    segs = (pl.read_parquet(t / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class", "internal", "left_token", "right_token"])
            .filter(pl.col("source") == "hifi").drop("source").with_columns(pl.col("start", "end").cast(pl.Int64)))
    rs = pl.read_parquet(t / "read_summary.parquet", columns=["zm", "source", "read_len", "primary_pattern", "skera_passing"]).filter(pl.col("source") == "hifi")
    sk = (pl.read_parquet(args.segments, columns=["zm", "di", "qs", "qe", "dl", "dr"]) if args.segments
          else read_sreads(Path(args.sreads), threads=4))
    sk = sk.with_columns(pl.col("zm").cast(pl.UInt32), pl.col("qs", "qe", "dl", "dr").cast(pl.Int64))
    st = single_tso_tso_mirrors(segs)
    summary = {"label": args.label, "single_tso_tso_mirror_reads": st.height}
    if args.non_passing:
        lens = dict(zip(rs["zm"].to_list(), rs["read_len"].to_list()))
        nonpass, full = set(), 0
        for r in iter_bam(args.non_passing):
            nonpass.add(r.get_tag("zm"))
            full += lens.get(r.get_tag("zm")) == r.query_length
        summary |= {"non_passing_records": len(nonpass), "non_passing_records_full_read_length": full}
        n_np = st.filter(pl.col("zm").is_in(pl.Series(list(nonpass), dtype=pl.UInt32))).height
    else:
        n_np = st.join(rs.select("zm", "skera_passing"), on="zm").filter(~pl.col("skera_passing")).height
    summary |= {"mirror_reads_non_passing": n_np, "mirror_reads_non_passing_share": n_np / st.height}

    cls = mirror_arm_classes(segs, sk).rename({"cls": "class"})
    reads, orientation = read_classes(segs, st, sk)
    shares(reads.with_columns(pl.lit("all").alias("set")), ["set"], "read_class").drop("set").write_csv(out / f"{args.label}_reads.tsv", separator="\t")
    shares(reads, ["letter"], "read_class").write_csv(out / f"{args.label}_reads_by_letter.tsv", separator="\t")
    orientation.write_csv(out / f"{args.label}_orientation.tsv", separator="\t")

    comp = rs.filter(pl.col("primary_pattern") == "complete").select("zm")
    if args.baseline_fraction < 1:
        comp = comp.filter(pl.col("zm").map_elements(lambda z: in_hash_sample(z, args.baseline_fraction), return_dtype=pl.Boolean))
    ccls = (molecule_coverage(segs.join(comp, on="zm"), sk.filter(pl.col("zm").is_in(comp["zm"]))).filter(USABLE)
            .select(pl.lit("complete_array").alias("region"), pl.col("cls").alias("class")))
    mol = pl.concat([shares(cls.filter(pl.col("region").is_in(["before", "after"])), ["region"], "class"),
                     shares(cls.with_columns(pl.lit("all_mirror_arms").alias("region")), ["region"], "class"),
                     shares(ccls, ["region"], "class")])
    mol.write_csv(out / f"{args.label}_molecules.tsv", separator="\t")
    summary |= {"proper_molecules_in_mirror_reads": cls.height, "clean_share_mirror_reads": float((cls["class"] == "CLEAN").mean()),
                "complete_reads_baseline": comp.height, "clean_share_complete_arrays": float((ccls["class"] == "CLEAN").mean())}

    if args.found_adapters:
        feats = pl.read_parquet(t / "read_features.parquet").filter(pl.col("source") == "hifi")
        table, counts = found_adapters(feats, st, args.found_adapters)
        table.write_csv(out / f"{args.label}_found_adapters.tsv", separator="\t")
        summary |= counts
    if args.ligations:
        lig = pl.read_csv(args.ligations)
        summary |= {"ligations_same_letter": int(lig.filter(pl.col("adapter_1") == pl.col("adapter_2"))["ligations"].sum()),
                    "ligations_total": int(lig["ligations"].sum())}
    (out / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
