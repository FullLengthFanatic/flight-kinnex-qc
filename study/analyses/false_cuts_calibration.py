#!/usr/bin/env python3
"""Where skera cuts without an adapter, on the calibration set.

In skera-passing HiFi reads, the gaps between consecutive skera S-reads that hold none of our adapters within 5 bp
(skera.false_cuts). For each such cut: the letter skera assigned (dr of the S-read before it), the closest Kinnex
adapter to the cut sequence (5 bp either side; edlib, both strands, edit distance; a tie goes to the earlier letter),
and whether a CTCCTCC-type repeat lies within 10 bp. Also our adapters that sit inside an S-read in the reads with at
least one such cut (skera.adapters_inside_sreads with the strict rule, qs + 5 < start and end < qe - 5; an adapter
counts once): edit distance and whether a TSO or Read1 hit lies within 4 bp.
Inputs: a truth directory with read_summary, read_features and segments (skera S-reads) parquets, and the HiFi BAM of
the same reads. Writes skera_false_cuts_summary.tsv, skera_false_cuts_by_letter.tsv (cuts by the letter skera
assigned, most first, ties in letter order) and skera_false_cuts_by_distance.tsv (closest adapter and its edit distance
to the cut sequence) into --out.
Usage: study/analyses/false_cuts_calibration.py --truth <dir> --hifi <bam> --out <dir>
"""
import argparse
from pathlib import Path

import edlib
import polars as pl

from kinnex_qc.io import iter_bam
from kinnex_qc.patterns import MAS, MAS_ORDER, revcomp
from kinnex_qc.skera import adapters_inside_sreads, false_cuts

REPEATS = ("CTCCTCCTCC", "GGAGGAGGAGG")
CUT_FLANK = 5          # bp either side of the cut searched for the closest adapter
REPEAT_FLANK = 10      # bp either side of the cut searched for a repeat
PRIMER_GAP = 4         # an inside adapter is flanked by a primer when a TSO or Read1 hit starts or ends this close to it


def closest_adapter(seq: str) -> tuple[str, int]:
    """The Kinnex adapter (either strand) with the fewest edits to any part of seq, and that edit count; a tie goes to
    the earlier letter."""
    d, letter = min((edlib.align(p, seq, mode="HW", task="distance")["editDistance"], name)
                    for name in MAS_ORDER for p in (MAS[name], revcomp(MAS[name])))
    return letter, d


def cut_table(fc: pl.DataFrame, seqs: dict[int, str]) -> pl.DataFrame:
    """Per false cut (skera.false_cuts' rows): the letter skera assigned, the closest adapter and its edits, and whether
    a repeat lies near the cut."""
    rows = []
    for zm, e1, q2, dr in fc.select("zm", "cut_start", "cut_end", "dr_before").iter_rows():
        seq = seqs[zm]
        best, d = closest_adapter(seq[max(0, e1 - CUT_FLANK): q2 + CUT_FLANK])
        near = seq[max(0, e1 - REPEAT_FLANK): q2 + REPEAT_FLANK]
        rows.append((zm, MAS_ORDER[dr], best, d, any(m in near for m in REPEATS)))
    return pl.DataFrame(rows, schema={"zm": pl.UInt32, "letter": pl.Utf8, "closest": pl.Utf8, "edits": pl.Int64, "repeat": pl.Boolean},
                        orient="row")


def inside_table(sreads: pl.DataFrame, features: pl.DataFrame, zms: pl.Series) -> pl.DataFrame:
    """Our MAS adapters strictly inside an S-read of the ZMWs zms, one row per adapter, with edits and whether a TSO or
    Read1 hit flanks it."""
    inside = (adapters_inside_sreads(sreads, features, strict=True).filter(pl.col("zm").is_in(zms))
              .unique(["zm", "name", "strand", "start", "end", "edits"]))
    primers = (features.filter(pl.col("kind").is_in(["TSO", "R1"])).select("zm", pl.col("start").cast(pl.Int64).alias("ps"),
                                                                           pl.col("end").cast(pl.Int64).alias("pe")))
    flank = ((pl.col("ps") - pl.col("end")).abs() <= PRIMER_GAP) | ((pl.col("start") - pl.col("pe")).abs() <= PRIMER_GAP)
    return (inside.join(primers, on="zm", how="left")
            .group_by("zm", "name", "strand", "start", "end", "edits").agg(flank.any().alias("flanked")))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="directory with read_summary, read_features and segments parquets")
    ap.add_argument("--hifi", required=True, help="HiFi BAM of the same reads")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)
    t, out = Path(args.truth), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    passing = (pl.read_parquet(t / "read_summary.parquet", columns=["zm", "source", "skera_passing"])
               .filter((pl.col("source") == "hifi") & pl.col("skera_passing"))["zm"])
    feats = pl.read_parquet(t / "read_features.parquet").filter((pl.col("source") == "hifi") & pl.col("zm").is_in(passing))
    sk = pl.read_parquet(t / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"]).filter(pl.col("zm").is_in(passing))
    fc = false_cuts(sk, feats)
    cut_zms = set(fc["zm"].to_list())
    seqs = {r.get_tag("zm"): r.query_sequence for r in iter_bam(args.hifi, keep=cut_zms.__contains__)}
    cuts = cut_table(fc, seqs)
    inside = inside_table(sk, feats, fc["zm"].unique())

    closest_d = cuts.filter(pl.col("closest") == "D")
    pl.DataFrame([{"skera_passing_reads": passing.len(), "reads_with_cut_without_adapter": len(cut_zms), "cuts": cuts.height,
                   "cuts_called_D_by_skera": cuts.filter(pl.col("letter") == "D").height, "cuts_closest_to_D": closest_d.height,
                   "cuts_closest_to_D_3_to_5_edits": closest_d.filter(pl.col("edits").is_between(3, 5)).height,
                   "cuts_within_2_edits_of_any_adapter": cuts.filter(pl.col("edits") <= 2).height,
                   "cuts_with_ctcctcc_repeat": cuts.filter(pl.col("repeat")).height, "our_adapters_inside_sreads": inside.height,
                   "inside_le1_edit_with_primer": inside.filter((pl.col("edits") <= 1) & pl.col("flanked")).height}]) \
        .write_csv(out / "skera_false_cuts_summary.tsv", separator="\t")
    (cuts.group_by("letter").len().rename({"letter": "letter_skera_assigned", "len": "cuts"})
     .sort(["cuts", "letter_skera_assigned"], descending=[True, False])
     .write_csv(out / "skera_false_cuts_by_letter.tsv", separator="\t"))
    (cuts.group_by("closest", "edits").len().rename({"closest": "closest_adapter", "len": "cuts"}).sort("closest_adapter", "edits")
     .write_csv(out / "skera_false_cuts_by_distance.tsv", separator="\t"))


if __name__ == "__main__":
    main()
