#!/usr/bin/env python3
"""Is "S-read with a TSO at both ends" a TSO-TSO signal? The skera 1.2.0 S-reads of the calibration set (the version
behind the public ArgenTag S-reads), first 200,000 S-reads, flagged when the TSO core lies within 80 bp of each end (up
to 3 edits, either strand: kinnex_qc.scan.best_hit with the package's TSO pattern). Each flagged S-read is placed on our
full-read molecules (same ZMW, overlapping coordinates; qs and qe are documented as parent-read coordinates,
https://skera.how/read-segments.html).

Inputs, in the calibration folder: nd_first200k.skera120.bam (skera_split.py) and nd_first200k_truth/annot_segments.parquet
and read_summary.parquet (build_calibration_set.py).
Writes tso_end_validation.tsv into --out (default: the calibration folder).
Usage, from the repository root:
  study/calibration/tso_end_validation.py [--calibration-dir DIR] [--out DIR]
--calibration-dir defaults to calibration_dir of study/study.toml and study.local.toml.
"""
import argparse
import sys
from pathlib import Path

import polars as pl

from kinnex_qc.io import iter_bam
from kinnex_qc.patterns import primer_patterns
from kinnex_qc.scan import best_hit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config  # noqa: E402

TSO = next(p for p in primer_patterns() if p.kind == "TSO")     # the TSO core, up to 3 edits
N, W = 200_000, 80


def has_tso(s: str) -> bool:
    return any(best_hit(s, TSO, strand, 0, len(s)) for strand in "+-")


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration-dir", default=config.get("calibration_dir"), help="calibration folder (build_calibration_set.py)")
    ap.add_argument("--out", help="output directory (default: the calibration folder)")
    args = ap.parse_args(argv)
    if not args.calibration_dir or args.calibration_dir.startswith("<"):
        ap.error("no calibration folder: give --calibration-dir or set calibration_dir in study/study.local.toml")
    here = Path(args.calibration_dir)
    out = Path(args.out) if args.out else here
    out.mkdir(parents=True, exist_ok=True)
    nd = here / "nd_first200k_truth"
    flagged = []
    for r in iter_bam(here / "nd_first200k.skera120.bam", max_records=N):
        s = r.query_sequence
        if has_tso(s[:W]) and has_tso(s[-W:]):
            flagged.append((r.get_tag("zm"), r.get_tag("qs"), r.get_tag("qe")))
    fl = pl.DataFrame(flagged, schema=["zm", "qs", "qe"], orient="row").with_columns(pl.col("zm").cast(pl.UInt32))
    seg = pl.read_parquet(nd / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class"]).filter(pl.col("source") == "hifi")
    rs = pl.read_parquet(nd / "read_summary.parquet", columns=["zm", "source", "primary_pattern"]).filter(pl.col("source") == "hifi").drop("source")
    hit = (fl.join(seg, on="zm", how="left").filter(pl.col("start").is_null() | ((pl.col("start") < pl.col("qe")) & (pl.col("end") > pl.col("qs"))))
           .group_by("zm", "qs", "qe").agg((pl.col("end_class") == "tso_tso").any().alias("touches_tso_tso")).join(rs, on="zm", how="left"))
    t = hit.group_by("touches_tso_tso", "primary_pattern").len().sort("len", descending=True).with_columns(
        (pl.col("len") / N).alias("share_of_sreads_checked"), pl.lit(N).alias("sreads_checked"))
    t.write_csv(out / "tso_end_validation.tsv", separator="\t")
    print(f"{fl.height} flagged of {N}")
    print(t)


if __name__ == "__main__":
    main()
