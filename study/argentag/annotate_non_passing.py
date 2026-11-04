#!/usr/bin/env python3
"""Annotate the parent reads of the ArgenTag 8-fold cell that skera did not pass with the study's stage 02 function
(kinnex_qc.runner.annotate_batch): adapter and primer search, adapter chain and pattern (complete, truncated, mirror,
...), molecules and their end classes.

Input: data/segmented.non_passing.bam in the ArgenTag folder, skera's file of the reads it did not pass for this cell.
It is not public; ArgenTag shared it with us on 2026-10-05, and it must not be redistributed. Per skera's documentation
(https://skera.how/output.html) this BAM "contains the bam records that have adapters found in non-sequential order",
so every read with a mirror junction is in it unless skera passed it.
Writes read_features, read_summary and annot_segments parquets (the study's schemas, source hifi, skera_passing false)
into <out>/full_reads/ (--out defaults to <argentag_dir>/analysis).
Usage, from the repository root:
  study/argentag/annotate_non_passing.py [--argentag-dir DIR] [--out DIR] [--processes N]
--argentag-dir and --processes default to argentag_dir and processes of study/study.toml and study.local.toml.
"""
import argparse
import sys
from pathlib import Path

from kinnex_qc.io import iter_bam
from kinnex_qc.parallel import ParquetBatchWriter, batched, bounded_imap, process_pool
from kinnex_qc.runner import BATCH, annotate_batch
from kinnex_qc.schemas import ANNOT_SEGMENTS, READ_FEATURES, READ_SUMMARY

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config  # noqa: E402


def records(path: Path):
    for r in iter_bam(path):
        yield (r.get_tag("zm"), "hifi", r.query_sequence, r.get_tag("np"), r.get_tag("rq"), False)


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--argentag-dir", default=config.get("argentag_dir"), help="ArgenTag folder, holding data/ and analysis/")
    ap.add_argument("--out", help="output directory (default: <argentag-dir>/analysis)")
    ap.add_argument("--processes", type=int, default=config["processes"])
    args = ap.parse_args(argv)
    if not args.argentag_dir or args.argentag_dir.startswith("<"):
        ap.error("no ArgenTag folder: give --argentag-dir or set argentag_dir in study/study.local.toml")
    arg = Path(args.argentag_dir)
    out = (Path(args.out) if args.out else arg / "analysis") / "full_reads"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with process_pool(args.processes) as pool, \
            ParquetBatchWriter(out / "read_features.parquet", READ_FEATURES) as wf, \
            ParquetBatchWriter(out / "read_summary.parquet", READ_SUMMARY) as ws, \
            ParquetBatchWriter(out / "annot_segments.parquet", ANNOT_SEGMENTS) as wa:
        src = records(arg / "data" / "segmented.non_passing.bam")
        for feats, summ, segs in bounded_imap(pool, annotate_batch, batched(src, BATCH), 4 * args.processes):
            wf.write(feats)
            ws.write(summ)
            wa.write(segs)
            n += len(summ["zm"])
            if n % 20_000 < BATCH:
                print(f"{n:,} reads", file=sys.stderr, flush=True)
    print(f"done: {n:,} reads", file=sys.stderr)


if __name__ == "__main__":
    main()
