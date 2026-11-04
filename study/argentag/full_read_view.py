#!/usr/bin/env python3
"""ArgenTag 8-fold, from the parent reads: mirror arrays, their junction molecules, TSO-TSO molecules, and how well the
unwritten-head flag on S-reads alone finds mirror reads in this library.

Inputs, in --out, the run's ArgenTag folder: full_reads/ (annotate_non_passing.py: the study's stage 02 on every read
skera did not pass) and argentag_sread_coords.parquet (sread_coords.py: zm, qs, qe, dl, dr of every public S-read).
The cell has 1,964,147 HiFi reads (Table 1 of the ArgenTag/PacBio technical note); observed: the ZMWs with S-reads and
the non-passing ZMWs together are exactly these 1,964,147. Per skera's documentation (https://skera.how/output.html) a
passing read has its adapters in sequential order, so it holds no mirror junction unless skera passed a read that has
one (observed in BD144_ND: rare for TSO-TSO mirrors).
Molecules per 1,000: our molecules in non-passing reads plus skera's S-reads of passing reads as the denominator.
The mirror junctions, their letters and the mirror size rule come from kinnex_qc.metrics, the unwritten-head flag from
kinnex_qc.skera.unwritten_head, usable molecules from kinnex_qc.molecules.USABLE.
Writes full_read_summary.tsv, full_read_patterns.tsv, full_read_junctions.tsv, full_read_mirror_rule.tsv and
flag_validation.tsv (aggregate tables) into --out.
Usage, from the repository root:
  study/argentag/full_read_view.py --out DIR
"""
import argparse
from pathlib import Path

import polars as pl

from kinnex_qc import metrics
from kinnex_qc.molecules import USABLE
from kinnex_qc.skera import unwritten_head


HIFI = 1_964_147     # HiFi reads of the ArgenTag 8-fold cell, Table 1 of the ArgenTag/PacBio technical note


def tables(rs: pl.DataFrame, seg: pl.DataFrame, coords: pl.DataFrame, hifi: int = HIFI) -> dict[str, pl.DataFrame]:
    """The five tables, by output name (without .tsv), from read_summary and annot_segments of the non-passing reads
    and the S-read coordinates of the whole cell (zm as Int64)."""
    non_passing = rs["zm"].cast(pl.Int64)
    passing_sreads = coords.filter(~pl.col("zm").is_in(non_passing)).height

    # patterns among non-passing reads, as shares of the whole cell
    patterns = rs.group_by("primary_pattern").agg(pl.len().alias("reads"), pl.col("read_len").median().alias("median_read_len")) \
        .with_columns((pl.col("reads") / hifi).alias("share_of_hifi_zmws")).sort("reads", descending=True)

    # junction molecules: same letter on both sides, opposite orientation
    junctions = metrics.junction_table(seg)
    shares = metrics.mirror_junction_shares(seg)
    single = metrics.mirror_size_reads(seg, rs)
    rule = metrics.mirror_size_rule(seg, rs)
    mirror_reads = rs.filter(pl.col("primary_pattern") == "mirror")

    # molecules per 1,000: ours in non-passing reads, skera's S-reads for passing reads
    mol_np = seg.height
    classes = dict(seg.group_by("end_class").len().iter_rows())
    denominator = mol_np + passing_sreads

    # the unwritten-head flag against the parent reads
    truth = set(mirror_reads["zm"].to_list())
    z = unwritten_head(coords).with_columns(pl.col("zm").is_in(mirror_reads["zm"].cast(pl.Int64)).alias("mirror"))
    fl, qa = z.filter(pl.col("flag")), z.filter(pl.col("q_not_a"))
    n_mirror = len(truth)
    validation = pl.DataFrame([{
        "hifi_zmws": hifi, "mirror_reads": n_mirror, "flagged": fl.height, "flag_share": fl.height / hifi,
        "flag_ppv": fl["mirror"].mean(), "flag_sensitivity": fl["mirror"].sum() / n_mirror,
        "flag_non_mirror_share": (~fl["mirror"]).sum() / hifi, "q_not_a": qa.height, "q_not_a_ppv": qa["mirror"].mean(),
        "mirror_without_sreads": len(truth - set(coords["zm"].unique().to_list()))}])

    # what mirror arrays cost: usable molecules (proper, not fused) per mirror read against S-reads per passing read
    usable = seg.filter(USABLE).group_by("zm").len().rename({"len": "usable"})
    nsk = coords.group_by("zm").len().rename({"len": "sreads"})
    mr = (mirror_reads.select(pl.col("zm").cast(pl.Int64)).join(usable.with_columns(pl.col("zm").cast(pl.Int64)), on="zm", how="left")
          .join(nsk, on="zm", how="left").fill_null(0))
    full = nsk.filter(~pl.col("zm").is_in(non_passing))["sreads"].mean()
    cost = {"usable_per_mirror_read": mr["usable"].mean(), "sreads_per_mirror_read": mr["sreads"].mean(), "sreads_per_passing_read": full,
            "mirror_shortfall_ours_per_hifi_zmw": n_mirror / hifi * (full - mr["usable"].mean()),
            "mirror_shortfall_sreads_per_hifi_zmw": n_mirror / hifi * (full - mr["sreads"].mean()),
            "proper_share_of_molecules": classes.get("proper", 0) / mol_np}

    summary = pl.DataFrame([{
        **cost,
        "hifi_zmws": hifi, "non_passing_reads": rs.height, "non_passing_share": rs.height / hifi,
        "mirror_reads": n_mirror, "mirror_share": n_mirror / hifi,
        "mirror_reads_median_len": mirror_reads["read_len"].median(),
        "mirror_junctions": shares["mirror_junctions"], "junctions_tso_tso_share": shares["mirror_junction_tso_tso_share"],
        "junctions_r1_r1_share": shares["mirror_junction_r1_r1_share"],
        "junctions_at_A_tso_tso": junctions.filter((pl.col("letter") == "A") & (pl.col("molecule") == "TSO-TSO"))["junctions"].sum(),
        "single_tt_reads": single.height, "single_tt_matching_rule": (single["molecules"] == single["predicted"]).mean(),
        "molecules_denominator": denominator, "molecules_in_non_passing": mol_np, "sreads_of_passing_reads": passing_sreads,
        "tso_tso_per_1000": 1000 * classes.get("tso_tso", 0) / denominator, "r1_r1_per_1000": 1000 * classes.get("r1_r1", 0) / denominator,
        "tso_tso_molecules": classes.get("tso_tso", 0), "r1_r1_molecules": classes.get("r1_r1", 0)}])
    return {"full_read_summary": summary, "full_read_patterns": patterns, "full_read_junctions": junctions,
            "full_read_mirror_rule": rule, "flag_validation": validation}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="the run's ArgenTag folder (never argentag_dir): holds the tables of sread_coords.py and annotate_non_passing.py")
    args = ap.parse_args(argv)
    out = Path(args.out)
    rs = pl.read_parquet(out / "full_reads" / "read_summary.parquet", columns=["zm", "read_len", "chain", "primary_pattern", "n_mas"])
    seg = pl.read_parquet(out / "full_reads" / "annot_segments.parquet",
                          columns=["zm", "start", "end", "left_token", "right_token", "end_class", "internal"])
    coords = pl.read_parquet(out / "argentag_sread_coords.parquet")
    t = tables(rs, seg, coords)
    for name, frame in t.items():
        frame.write_csv(out / f"{name}.tsv", separator="\t")
    with pl.Config(tbl_width_chars=200, tbl_cols=30, float_precision=4):
        print(t["full_read_summary"].transpose(include_header=True))
        print(t["full_read_patterns"])
        print(t["full_read_junctions"])
        print(t["full_read_mirror_rule"])
        print(t["flag_validation"].transpose(include_header=True))


if __name__ == "__main__":
    main()
