#!/usr/bin/env python3
"""Reads where skera's cuts and ours disagree, for a human to judge.

Among skera-passing HiFi reads, a read disagrees when some gap between consecutive S-reads holds none of our adapters
within 5 bp (skera.false_cuts), or one of our adapters lies inside an S-read by more than 5 bp (skera.adapters_inside_sreads,
the strict rule qs + 5 < start and end < qe - 5, as in the skera gate). From a random subset of skera-passing reads
(--pool, fixed seed), draws --n disagreeing reads (fixed seed) and writes the automatic description of each
disagreement to discordant_reads_auto.tsv, every disagreeing read of the subset to discordant_all_in_pool.tsv, the drawn
ZMWs to discordant_reads_zmws.tsv and the counts to discordant_summary.json. The review page discordant_reads.html
draws the drawn reads in the order of discordant_reads_zmws.tsv, five per figure, with the report's read-track code
(study/report/read_tracks.py): our adapters, primers and molecules, and skera's S-read boundaries as ticks; plotly is
inlined, so the page opens without a network.
In discordant_summary.json, our_adapters_inside_sreads counts one item per ';'-separated entry of each disagreeing read's
inside_adapters, so a disagreeing read with no adapter of ours inside an S-read (only a gap without our adapter) counts
one empty item: the value is the number of adapters inside S-reads plus the number of such reads (3,880 = 3,864 adapters
+ 16 reads on BD144_ND). It is kept this way to equal v10; quote it as items, and use inside_exact and inside_adapter_D
(which skip empty items) for adapter counts.
Usage: study/analyses/discordant_reads.py --sample-dir <pipeline sample dir> --out <dir> [--n 30] [--pool 40000]
       [--seed 11]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import polars as pl

from kinnex_qc.skera import adapters_inside_sreads, false_cuts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "report"))
from read_tracks import read_track_figure, tracks  # noqa: E402


def disagreements(sk: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """One row per read of sk that disagrees: its gaps without our adapter and our adapters inside an S-read, counted and
    listed. Reads in the order v10 drew from: by the start of the read's first S-read, then by S-read table order."""
    gaps = (false_cuts(sk, feats).group_by("zm", maintain_order=True)
            .agg(pl.len().alias("skera_gaps_without_our_adapter"),
                 pl.format("{}-{}", "cut_start", "cut_end").str.join(";").alias("gap_positions")))
    inside = (adapters_inside_sreads(sk, feats, strict=True).sort("zm", "start", "di").group_by("zm", maintain_order=True)
              .agg(pl.len().alias("our_adapters_inside_an_sread"),
                   pl.format("{}{}@{}(edits {})", "name", pl.when(pl.col("strand") == "+").then(pl.lit("")).otherwise(pl.lit("'")),
                             "start", "edits").str.join(";").alias("inside_adapters")))
    order = sk.with_row_index("row").group_by("zm").agg(pl.col("qs").min(), pl.col("row").min())
    return (gaps.join(inside, on="zm", how="full", coalesce=True).join(order, on="zm").sort("qs", "row")
            .with_columns(pl.col("skera_gaps_without_our_adapter", "our_adapters_inside_an_sread").fill_null(0).cast(pl.Int64),
                          pl.col("inside_adapters", "gap_positions").fill_null(""))
            .select("zm", "skera_gaps_without_our_adapter", "our_adapters_inside_an_sread", "inside_adapters", "gap_positions"))


def summary(disc: pl.DataFrame, n_pool: int, drawn: int) -> dict:
    """The counts of discordant_summary.json. As in v10, each disagreeing read's inside_adapters is split on ';', so a
    read without an adapter inside an S-read (an empty list) counts one item in our_adapters_inside_sreads."""
    ins = (disc.filter(pl.col("inside_adapters").is_not_null()).with_columns(pl.col("inside_adapters").str.split(";")).explode("inside_adapters")
           .with_columns(pl.col("inside_adapters").str.extract(r"^([A-HQ])").alias("letter"),
                         pl.col("inside_adapters").str.extract(r"edits (\d)").cast(pl.Int8).alias("edits")))
    shifted = disc.filter((pl.col("skera_gaps_without_our_adapter") == 1) & (pl.col("our_adapters_inside_an_sread") == 1)).height
    return {"pool_skera_passing_reads": n_pool, "disagreeing_reads": disc.height, "one_shifted_cut": shifted,
            "our_adapters_inside_sreads": ins.height, "inside_exact": int((ins["edits"] == 0).sum()),
            "inside_adapter_D": int((ins["letter"] == "D").sum()), "drawn_for_review": drawn}


def review_page(tr: list[dict]) -> str:
    """The HTML page of the drawn reads' tracks, five reads per figure, plotly inlined once."""
    figs = []
    for i in range(0, len(tr), 5):
        f = read_track_figure(tr[i:i + 5], f"Reads {i + 1} to {min(i + 5, len(tr))} of {len(tr)}: ticks are skera S-read boundaries")
        figs.append(f.to_html(full_html=False, include_plotlyjs=i == 0))
    return ("<html><head><meta charset='utf-8'><title>Discordant reads</title></head><body>"
            "<h2>Reads where skera's cuts and our adapters disagree</h2><p>Judge each read before "
            "opening discordant_reads_auto.tsv. Hover shows edit distances.</p>" + "".join(figs) + "</body></html>")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", required=True, help="pipeline output directory of one sample")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=30, help="disagreeing reads drawn for review")
    ap.add_argument("--pool", type=int, default=40_000, help="skera-passing reads drawn first, the subset searched")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    d, out = Path(args.sample_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reads = pl.read_parquet(d / "read_summary.parquet", columns=["zm", "source", "read_len", "chain", "primary_pattern", "skera_passing"])
    passing = reads.filter((pl.col("source") == "hifi") & pl.col("skera_passing"))["zm"].to_list()
    pool = pl.Series(sorted(random.Random(args.seed).sample(passing, min(args.pool, len(passing)))), dtype=pl.UInt32)
    feats = pl.scan_parquet(d / "read_features.parquet").filter((pl.col("source") == "hifi") & pl.col("zm").is_in(pool)).collect()
    sk = pl.read_parquet(d / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"]).filter(pl.col("zm").is_in(pool))
    disc = disagreements(sk, feats)
    n_pool = sk["zm"].n_unique()
    print(f"{disc.height:,} of {n_pool:,} skera-passing reads disagree ({disc.height / n_pool:.3%})")
    pick = sorted(random.Random(args.seed + 1).sample(disc["zm"].to_list(), min(args.n, disc.height)))
    segs = pl.read_parquet(d / "annot_segments.parquet", columns=["zm", "source", "start", "end", "pos", "end_class", "internal",
                                                                 "polyT_len", "tso_side"]).filter(pl.col("zm").is_in(pick))
    tr = tracks(pick, reads, segs, feats.filter(pl.col("zm").is_in(pick)), sk.select("zm", "qs", "qe"))
    (out / "discordant_reads.html").write_text(review_page(tr), encoding="utf-8")
    seq_note = reads.filter(pl.col("zm").is_in(pick)).select("zm", "read_len", "chain", "primary_pattern")
    disc.filter(pl.col("zm").is_in(pick)).join(seq_note, on="zm").sort("zm").write_csv(out / "discordant_reads_auto.tsv", separator="\t")
    pl.DataFrame({"zm": pick}).write_csv(out / "discordant_reads_zmws.tsv", separator="\t")
    disc.write_csv(out / "discordant_all_in_pool.tsv", separator="\t")
    (out / "discordant_summary.json").write_text(json.dumps(summary(disc, n_pool, len(pick)), indent=1))


if __name__ == "__main__":
    main()
