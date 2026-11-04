#!/usr/bin/env python3
"""Anatomy of skera's false cuts in complete arrays that skera passes, for one library.

False cut: a gap between two consecutive S-reads with no adapter of ours within 5 bp (skera.false_cuts). For each: the
molecule of ours that contains the cut; the adapter of ours left inside one of the two S-reads (missed adapter, the one
nearest the cut: skera.skipped_adapters, inclusive inside rule start >= qs + 5 and end <= qe - 5); which S-read is a
fragment of the cut molecule and which holds the rest of it fused to the next molecule; their end classes (stage 01)
and fates (stage 05); whether the fused S-read's far end sits at a real adapter of ours; which end of the cut molecule
(TSO or Read1) the fragment keeps, by read orientation (S-read adapters ascending along the read or not).
Writes <label>_false_cuts.json and <label>_false_cut_classes.tsv into --out.
Usage: study/analyses/false_cut_anatomy.py --sample-dir <pipeline sample dir> --label <name> --out <dir>
"""
import argparse
import json
from pathlib import Path

import polars as pl

from kinnex_qc.patterns import END_WINDOW, MAS_ORDER
from kinnex_qc.skera import TOL, false_cuts, skipped_adapters

FAR_SLACK = 25          # an S-read's far end is at a real adapter when it is within TOL bp of the adapter's near edge or this many of its far edge
SREAD_COLS = ["qs", "qe", "dl", "dr", "end_class", "lima_pass", "lima_pair", "refine_asrun", "mapped_fltnc"]


def false_cut_frame(sk: pl.DataFrame, feats: pl.DataFrame, mols: pl.DataFrame) -> pl.DataFrame:
    """One row per false cut: the S-read before it (columns of SREAD_COLS) and after it (the same with suffix _2), the
    cut's centre, the molecule containing it (m_start, m_end, m_class), the missed adapter (missed, in_sread, dist), the
    primers at the two ends of that molecule (left, right) and whether the S-reads' outer ends sit at our adapters
    (at_s1_start, at_s2_end)."""
    fc = false_cuts(sk, feats)
    before = sk.select("zm", pl.col("di").alias("di_before"), *SREAD_COLS)
    after = sk.select("zm", pl.col("di").alias("di_after"), *[pl.col(c).alias(f"{c}_2") for c in SREAD_COLS])
    missed = skipped_adapters(fc, sk, feats).select("zm", "cut_start", "missed", "in_sread", "dist")
    fc = (fc.join(before, on=["zm", "di_before"]).join(after, on=["zm", "di_after"]).join(missed, on=["zm", "cut_start"], how="left")
          .with_row_index("gid").with_columns(((pl.col("qe") + pl.col("qs_2")) // 2).alias("cut")))
    # molecule containing the cut
    m = (fc.select("gid", "zm", "cut").join(mols, on="zm")
         .filter((pl.col("start") <= pl.col("cut")) & (pl.col("end") > pl.col("cut")))
         .group_by("gid").agg(pl.col("start").first().alias("m_start"), pl.col("end").first().alias("m_end"),
                              pl.col("end_class").first().alias("m_class")))
    fc = fc.join(m, on="gid", how="left")
    # primers at the two ends of the cut molecule
    prim = feats.filter(pl.col("kind").is_in(["TSO", "R1"]))
    ends = (fc.select("gid", "zm", "m_start", "m_end").join(prim, on="zm")
            .with_columns(pl.when((pl.col("start") - pl.col("m_start")).abs() <= END_WINDOW).then(pl.lit("left"))
                          .when((pl.col("end") - pl.col("m_end")).abs() <= END_WINDOW).then(pl.lit("right")).otherwise(None).alias("side"))
            .filter(pl.col("side").is_not_null()).group_by("gid", "side").agg(pl.col("kind").first())
            .pivot(on="side", index="gid", values="kind"))
    fc = fc.join(ends, on="gid", how="left")
    # far end of S2 / start of S1 at a real adapter?
    mas = feats.filter(pl.col("kind") == "MAS")
    far = (fc.select("gid", "zm", "qs", "qe_2").join(mas, on="zm")
           .with_columns((((pl.col("end") - pl.col("qs")).abs() <= TOL) | ((pl.col("start") - pl.col("qs")).abs() <= FAR_SLACK)).alias("at_s1_start"),
                         (((pl.col("start") - pl.col("qe_2")).abs() <= TOL) | ((pl.col("end") - pl.col("qe_2")).abs() <= FAR_SLACK)).alias("at_s2_end"))
           .group_by("gid").agg(pl.col("at_s1_start").any(), pl.col("at_s2_end").any()))
    return fc.join(far, on="gid", how="left")


def fragment_columns(fc: pl.DataFrame) -> pl.DataFrame:
    """The letter skera called, which S-read is the fragment (S2 when the missed adapter sits in S1, and the reverse),
    the cut's place in its molecule, and each fragment and fused S-read field taken from S1 or S2 accordingly."""
    fc = fc.with_columns(
        pl.col("dr").cast(pl.Int64).replace_strict(dict(enumerate(MAS_ORDER)), return_dtype=pl.Utf8).alias("skera_letter"),
        pl.when(pl.col("in_sread") == 1).then(pl.lit("S2")).when(pl.col("in_sread") == 2).then(pl.lit("S1")).otherwise(None).alias("fragment"),
        ((pl.col("cut") - pl.col("m_start")) / (pl.col("m_end") - pl.col("m_start"))).alias("cut_frac"))
    s1 = pl.col("fragment") == "S1"
    return fc.with_columns(
        pl.when(s1).then(pl.col("left")).when(pl.col("fragment") == "S2").then(pl.col("right")).alias("fragment_keeps"),
        pl.when(s1).then(pl.col("end_class")).when(pl.col("fragment") == "S2").then(pl.col("end_class_2")).alias("frag_class"),
        pl.when(s1).then(pl.col("end_class_2")).when(pl.col("fragment") == "S2").then(pl.col("end_class")).alias("fused_class"),
        pl.when(s1).then(pl.col("lima_pass")).otherwise(pl.col("lima_pass_2")).alias("frag_lima"),
        pl.when(s1).then(pl.col("lima_pass_2")).otherwise(pl.col("lima_pass")).alias("fused_lima"),
        pl.when(s1).then(pl.col("lima_pair")).otherwise(pl.col("lima_pair_2")).alias("frag_pair"),
        pl.when(s1).then(pl.col("lima_pair_2")).otherwise(pl.col("lima_pair")).alias("fused_pair"),
        pl.when(s1).then(pl.col("mapped_fltnc")).otherwise(pl.col("mapped_fltnc_2")).alias("frag_mapped"),
        pl.when(s1).then(pl.col("mapped_fltnc_2")).otherwise(pl.col("mapped_fltnc")).alias("fused_mapped"),
        pl.when(s1).then(pl.col("refine_asrun")).otherwise(pl.col("refine_asrun_2")).alias("frag_refine"),
        pl.when(s1).then(pl.col("refine_asrun_2")).otherwise(pl.col("refine_asrun")).alias("fused_refine"),
        pl.when(s1).then(pl.col("at_s2_end")).otherwise(pl.col("at_s1_start")).alias("fused_far_end_at_adapter"),
        (pl.col("dl") < pl.col("dr")).alias("ascending"))


def summary(label: str, target: pl.Series, sk: pl.DataFrame, fc: pl.DataFrame) -> dict:
    def share(expr):
        return float(fc.select(expr.mean()).item())

    with_missed = fc.filter(pl.col("missed").is_not_null())
    summ = {"label": label, "complete_passing_reads": target.len(), "false_cuts": fc.height, "false_cuts_per_read": fc.height / target.len(),
            "reads_with_false_cut": fc["zm"].n_unique() / target.len(),
            "skera_calls_D": share(pl.col("skera_letter") == "D"),
            "missed_adapter_found": share(pl.col("missed").is_not_null()),
            "missed_is_D": float(with_missed.select((pl.col("missed") == "D").mean()).item()),
            "missed_same_letter_as_skera_call": float(with_missed.select((pl.col("missed") == pl.col("skera_letter")).mean()).item()),
            "cut_inside_proper_molecule": share(pl.col("m_class") == "proper"),
            "median_cut_to_missed_adapter_bp": float(fc["dist"].median()),
            "fused_far_end_at_real_adapter": share(pl.col("fused_far_end_at_adapter").fill_null(False)),
            "fragment_after_cut_in_read_order": share(pl.col("fragment") == "S2"),
            "fragment_keeps_tso_when_ascending": float(fc.filter(pl.col("ascending") & pl.col("fragment_keeps").is_not_null())
                                                       .select((pl.col("fragment_keeps") == "TSO").mean()).item()),
            "fragment_keeps_r1_when_descending": float(fc.filter(~pl.col("ascending") & pl.col("fragment_keeps").is_not_null())
                                                        .select((pl.col("fragment_keeps") == "R1").mean()).item()),
            "cut_position_from_kept_end_median": float(fc.select(pl.when(pl.col("fragment") == "S1").then(pl.col("cut_frac"))
                                                                 .otherwise(1 - pl.col("cut_frac")).median()).item())}
    done = fc.filter(pl.col("fragment").is_not_null())
    summ["fates"] = done.select(pl.col("frag_lima", "frag_refine", "frag_mapped", "fused_lima", "fused_refine", "fused_mapped")
                                .cast(pl.Float64).mean()).to_dicts()[0]
    summ["baseline_proper_sreads"] = (sk.filter(pl.col("end_class") == "proper")
                                      .select(pl.col("lima_pass", "refine_asrun", "mapped_fltnc").cast(pl.Float64).mean()).to_dicts()[0])
    return summ


def classes(fc: pl.DataFrame) -> pl.DataFrame:
    return (pl.concat([fc.group_by(col).len().rename({col: "value", "len": "false_cuts"})
                       .with_columns(pl.lit(col).alias("field"), pl.col("value").cast(pl.Utf8))
                       for col in ("skera_letter", "missed", "fragment", "fragment_keeps", "frag_class", "fused_class", "frag_pair", "fused_pair")])
            .select("field", "value", "false_cuts").sort("field", "false_cuts", descending=[False, True]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", required=True, help="pipeline output directory of one sample")
    ap.add_argument("--label", required=True, help="name for the output files")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)
    o, out = Path(args.sample_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rs = (pl.read_parquet(o / "read_summary.parquet", columns=["zm", "source", "skera_passing", "primary_pattern"])
          .filter(pl.col("source") == "hifi"))
    target = rs.filter(pl.col("skera_passing") & (pl.col("primary_pattern") == "complete"))["zm"]
    sk = (pl.read_parquet(o / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr", "end_class"]).filter(pl.col("zm").is_in(target))
          .join(pl.read_parquet(o / "fate.parquet", columns=["zm", "di", "lima_pass", "lima_pair", "refine_asrun", "mapped_fltnc"]),
                on=["zm", "di"], how="left")
          .with_columns(pl.col("qs", "qe").cast(pl.Int64)))
    feats = (pl.read_parquet(o / "read_features.parquet", columns=["zm", "source", "kind", "name", "strand", "start", "end", "edits"])
             .filter((pl.col("source") == "hifi") & pl.col("zm").is_in(target)).drop("source")
             .with_columns(pl.col("start", "end").cast(pl.Int64)))
    mols = (pl.read_parquet(o / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class"])
            .filter((pl.col("source") == "hifi") & pl.col("zm").is_in(target)).drop("source")
            .with_columns(pl.col("start", "end").cast(pl.Int64)))
    fc = fragment_columns(false_cut_frame(sk, feats, mols))
    summ = summary(args.label, target, sk, fc)
    classes(fc).write_csv(out / f"{args.label}_false_cut_classes.tsv", separator="\t")
    (out / f"{args.label}_false_cuts.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
