#!/usr/bin/env python3
"""Calibration set: the first 200,000 HiFi reads of the BD144_ND cell, used where the same reads must be compared
between methods (skera 1.2.0 against 26.1.0, the TSO-at-both-ends check, the downstream test, skera's false cuts).
They are not a random sample of the cell.

1. nd_first200k.hifi.nomod.bam: the first 200,000 records of BD144_ND's HiFi BAM, base-modification tags (MM, ML)
   removed (observed: skera 1.2.0 stops with "cannot create std::vector larger than max_size()" on the first 50 of
   these records with the tags, and runs on the same records without them). Written only if absent.
2. nd_first200k_truth/: our full-read annotation of these reads with the study's stage 02 function
   (kinnex_qc.runner.annotate_batch: read_summary, annot_segments, read_features), plus the S-reads skera 26.1.0
   delivered for them (segments.parquet: zm, di, qs, qe, dl, dr) and skera's verdict (skera_passing: the ZMW is not in
   the delivered non-passing BAM). The segmented and non-passing BAMs are read from their first record of these ZMWs
   up to the first record past them: both files follow the HiFi BAM's ZMW order.
skera 1.2.0 (bioconda pbskera=1.2.0) then runs on the BAM of step 1 with the Kinnex 8-fold adapters: skera_split.py.
The BD144_ND row of --samples gives the input BAMs.
Usage, from the repository root:
  study/calibration/build_calibration_set.py [--samples PATH] [--out DIR] [--processes N]
--samples, --out and --processes default to samples, calibration_dir and processes of study/study.toml and
study.local.toml.
"""
import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pysam

from kinnex_qc.io import iter_bam
from kinnex_qc.parallel import ParquetBatchWriter, batched, bounded_imap, process_pool
from kinnex_qc.runner import BATCH, annotate_batch
from kinnex_qc.schemas import ANNOT_SEGMENTS, READ_FEATURES, READ_SUMMARY

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config, load_samples  # noqa: E402

SAMPLE = "BD144_ND"
N = 200_000


def same_zmws(records: Iterable[pysam.AlignedSegment], zms: set[int]) -> Iterator[pysam.AlignedSegment]:
    """The records whose ZMW is in zms, from the first such record up to the first record outside zms after it."""
    seen = False
    for r in records:
        if r.get_tag("zm") in zms:
            seen = True
            yield r
        elif seen:
            return


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=config["samples"], help="samples table")
    ap.add_argument("--out", default=config.get("calibration_dir"), help="calibration folder to write")
    ap.add_argument("--processes", type=int, default=config["processes"])
    args = ap.parse_args(argv)
    if not args.out or args.out.startswith("<"):
        ap.error("no calibration folder: give --out or set calibration_dir in study/study.local.toml")
    s = load_samples(args.samples)[SAMPLE]
    here = Path(args.out)
    here.mkdir(parents=True, exist_ok=True)
    bam = here / "nd_first200k.hifi.nomod.bam"
    if not bam.exists():
        records = iter_bam(s.hifi_bam, max_records=N)
        first = next(records)
        with pysam.AlignmentFile(str(bam), "wb", header=first.header) as out:
            for r in [first, *records]:
                for t in ("MM", "ML"):
                    if r.has_tag(t):
                        r.set_tag(t, None)
                out.write(r)
    reads = []
    with pysam.AlignmentFile(str(bam), "rb", check_sq=False) as f:
        for r in f:
            reads.append((r.get_tag("zm"), r.query_sequence, r.get_tag("np"), r.get_tag("rq")))
    zms = {zm for zm, *_ in reads}
    non_passing = {r.get_tag("zm") for r in same_zmws(iter_bam(s.non_passing_bam), zms)}
    cols = {k: [] for k in ("zm", "di", "qs", "qe", "dl", "dr")}
    for r in same_zmws(iter_bam(s.segmented_bam), zms):    # same ZMW order as the HiFi BAM
        for k in cols:
            cols[k].append(r.get_tag(k))
    out = here / "nd_first200k_truth"
    out.mkdir(exist_ok=True)
    pl.DataFrame({k: np.array(v, np.int64) for k, v in cols.items()}).with_columns(pl.col("zm").cast(pl.UInt32)) \
        .write_parquet(out / "segments.parquet")
    src = ((zm, "hifi", seq, np_, rq, zm not in non_passing) for zm, seq, np_, rq in reads)
    with process_pool(args.processes) as pool, \
            ParquetBatchWriter(out / "read_features.parquet", READ_FEATURES) as wf, \
            ParquetBatchWriter(out / "read_summary.parquet", READ_SUMMARY) as ws, \
            ParquetBatchWriter(out / "annot_segments.parquet", ANNOT_SEGMENTS) as wa:
        for feats, summ, segs in bounded_imap(pool, annotate_batch, batched(src, BATCH), 4 * args.processes):
            wf.write(feats)
            ws.write(summ)
            wa.write(segs)
    print(f"{len(reads):,} reads, {len(non_passing):,} non-passing, {len(cols['zm']):,} skera 26.1.0 S-reads", file=sys.stderr)


if __name__ == "__main__":
    main()
