#!/usr/bin/env python3
"""zm, qs, qe, dl and dr of every S-read in the public ArgenTag MDA-MB-453 8-fold S-read BAM, for the unwritten-head test
(kinnex_qc.skera.unwritten_head) of full_read_view.py and the report. qs and qe are documented as the S-read's start and
end in the parent read (https://skera.how/read-segments.html).

Input: data/segmented.bam in the ArgenTag folder, the public S-reads of the 8-fold cell (skera 1.2.0, as the BAM's @PG
line says; docs/data.md gives where to download it).
Writes argentag_sread_coords.parquet into --out (default <argentag_dir>/analysis): one pass over the 13 GB BAM, about
10 minutes.
Usage, from the repository root:
  study/argentag/sread_coords.py [--argentag-dir DIR] [--out DIR]
--argentag-dir defaults to argentag_dir of study/study.toml and study.local.toml.
"""
import argparse
import sys
from pathlib import Path

import polars as pl

from kinnex_qc.runner import read_sreads

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--argentag-dir", default=config.get("argentag_dir"), help="ArgenTag folder, holding data/ and analysis/")
    ap.add_argument("--out", help="output directory (default: <argentag-dir>/analysis)")
    args = ap.parse_args(argv)
    if not args.argentag_dir or args.argentag_dir.startswith("<"):
        ap.error("no ArgenTag folder: give --argentag-dir or set argentag_dir in study/study.local.toml")
    arg = Path(args.argentag_dir)
    out = Path(args.out) if args.out else arg / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    coords = read_sreads(arg / "data" / "segmented.bam", threads=4).select(pl.col("zm").cast(pl.Int64), "qs", "qe", "dl", "dr")
    coords.write_parquet(out / "argentag_sread_coords.parquet")
    print(coords.height, "S-reads")


if __name__ == "__main__":
    main()
