#!/usr/bin/env python3
"""Is the unwritten-head test the same under skera 1.2.0 (the version of the public ArgenTag S-reads) and 26.1.0 (the
version of the delivered BD144 S-reads; both as the @PG lines of their BAMs say)? Same 200,000 BD144_ND HiFi ZMWs,
both versions. The test is kinnex_qc.skera.unwritten_head: S-reads reach Q but not A, all in forward adapter order
(dl < dr), and the parent read before the first S-read (lowest qs) is at least 2 mean written molecules long; also the
share of ZMWs whose S-reads reach Q but not A. Truth: our mirror calls in nd_first200k_truth/ (build_calibration_set.py).

Inputs, in the calibration folder: nd_first200k.skera120.bam (skera_split.py), nd_first200k.hifi.nomod.bam, and
nd_first200k_truth/segments.parquet (the delivered skera 26.1.0 S-reads) and read_summary.parquet.
Writes head_test_skera_versions.tsv into --out.
Usage, from the repository root:
  study/calibration/head_version_check.py --out DIR [--calibration-dir DIR]
--calibration-dir defaults to calibration_dir of study/study.toml and study.local.toml. --out is required:
calibration_dir holds v10's calibration set, the reference the outputs are compared with, and is never written.
"""
import argparse
import sys
from pathlib import Path

import polars as pl

from kinnex_qc.io import iter_bam
from kinnex_qc.runner import read_sreads
from kinnex_qc.skera import unwritten_head

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration-dir", default=config.get("calibration_dir"), help="calibration folder (build_calibration_set.py)")
    ap.add_argument("--out", required=True, help="output directory (never calibration_dir, v10's reference set)")
    args = ap.parse_args(argv)
    if not args.calibration_dir or args.calibration_dir.startswith("<"):
        ap.error("no calibration folder: give --calibration-dir or set calibration_dir in study/study.local.toml")
    here = Path(args.calibration_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nd = here / "nd_first200k_truth"
    old = read_sreads(here / "nd_first200k.skera120.bam", threads=4).select("zm", "qs", "qe", "dl", "dr").cast(pl.Int64)
    zms = pl.Series("zm", [r.get_tag("zm") for r in iter_bam(here / "nd_first200k.hifi.nomod.bam")], dtype=pl.Int64)
    new = pl.read_parquet(nd / "segments.parquet", columns=["zm", "qs", "qe", "dl", "dr"]).cast(pl.Int64).filter(pl.col("zm").is_in(zms))
    truth = pl.read_parquet(nd / "read_summary.parquet", columns=["zm", "source", "primary_pattern"]).filter(pl.col("source") == "hifi").select(
        pl.col("zm").cast(pl.Int64), "primary_pattern")
    rows = []
    for name, sk in (("skera 1.2.0", old), ("skera 26.1.0", new)):
        z = unwritten_head(sk)
        q_not_a = z.filter(pl.col("q_not_a")).height
        flag = z.filter(pl.col("flag")).join(truth, on="zm", how="left")
        tp = flag.filter(pl.col("primary_pattern") == "mirror").height
        m = truth.filter(pl.col("zm").is_in(zms) & (pl.col("primary_pattern") == "mirror")).height
        rows.append({"skera": name, "hifi_zmws": len(zms), "sreads": sk.height, "q_not_a_share": q_not_a / len(zms),
                     "flagged": flag.height, "flagged_share": flag.height / len(zms),
                     "true_mirrors_flagged": tp, "ppv": tp / flag.height, "sensitivity": tp / m})
    t = pl.DataFrame(rows)
    t.write_csv(out / "head_test_skera_versions.tsv", separator="\t")
    print(t)


if __name__ == "__main__":
    main()
