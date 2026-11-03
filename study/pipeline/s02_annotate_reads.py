#!/usr/bin/env python3
"""Stage 02: index HiFi, fail and non-passing reads; annotate every HiFi read and a fail-read subsample.

Our segmentation replaces skera's for Q1 and Q2. Writes hifi_index, fail_index,
non_passing_index, read_features, read_summary and annot_segments parquet files.

Usage, from the repository root:
  study/pipeline/s02_annotate_reads.py --sample <name> [--samples PATH] [--out-root DIR] [--fraction F]
      [--fail-subsample N] [--processes N]
Defaults come from study/study.toml.
"""
import argparse
import itertools
import json
import sys
import time

import polars as pl

from config import load_config, load_samples, out_dir
from kinnex_qc.io import iter_bam
from kinnex_qc.parallel import ParquetBatchWriter, batched, bounded_imap, process_pool
from kinnex_qc.runner import annotate_batch
from kinnex_qc.sampling import in_hash_sample, keep_zmw
from kinnex_qc.schemas import ANNOT_SEGMENTS, READ_FEATURES, READ_INDEX, READ_SUMMARY

BATCH = 200


class Indexer:
    """Writes one READ_INDEX row per record and passes the records to annotate through."""

    def __init__(self, path, source, keep, skera_passing=lambda zm: True):
        self.writer = ParquetBatchWriter(path, READ_INDEX)
        self.source, self.keep, self.skera_passing = source, keep, skera_passing
        self.rows, self.n = {f.name: [] for f in READ_INDEX}, 0

    def feed(self, records):
        for r in records:
            zm, np_, rq, seq = r.get_tag("zm"), r.get_tag("np"), r.get_tag("rq"), r.query_sequence
            for k, v in (("zm", zm), ("read_len", len(seq)), ("np", np_), ("rq", rq)):
                self.rows[k].append(v)
            self.n += 1
            if len(self.rows["zm"]) >= 100_000:
                self.flush()
            if self.keep(zm):
                yield (zm, self.source, seq, np_, rq, self.skera_passing(zm))
        self.flush()
        self.writer.close()

    def flush(self):
        if self.rows["zm"]:
            self.writer.write(self.rows)
            self.rows = {f.name: [] for f in READ_INDEX}


def main() -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--samples", default=config["samples"], help="samples table")
    ap.add_argument("--out-root", default=config["out_root"], help="output root")
    ap.add_argument("--fraction", type=float, default=config["fraction"], help="hash sample of the cell's ZMWs; 1 reads them all")
    ap.add_argument("--fail-subsample", type=int, default=config["fail_subsample"], help="annotate 1 in N fail reads")
    ap.add_argument("--processes", type=int, default=config["processes"])
    args = ap.parse_args()

    sample = load_samples(args.samples)[args.sample]
    out = out_dir(args.out_root, sample.sample, args.fraction)
    keep = (lambda zm: in_hash_sample(zm, args.fraction)) if args.fraction < 1 else None
    modulo = args.fail_subsample
    t0 = time.time()

    # non-passing first: its ZMW set gives skera_passing for every HiFi read
    nonp = Indexer(out / "non_passing_index.parquet", "non_passing", lambda zm: False)
    for _ in nonp.feed(iter_bam(sample.non_passing_bam, keep=keep)):
        pass
    non_passing = set(pl.read_parquet(out / "non_passing_index.parquet", columns=["zm"])["zm"])
    hifi = Indexer(out / "hifi_index.parquet", "hifi", lambda zm: True, lambda zm: zm not in non_passing)
    fail = Indexer(out / "fail_index.parquet", "fail_sub", lambda zm: keep_zmw(zm, modulo))
    # A hash sample holds part of the HiFi ZMWs but indexes the whole fail BAM, so stage 06 knows the share
    # of the cell's fail reads it holds (all of them) and can weigh HiFi and fail reads to the cell.
    tasks = itertools.chain(
        hifi.feed(iter_bam(sample.hifi_bam, keep=keep)),
        fail.feed(iter_bam(sample.fail_bam)),
    )
    n = 0
    with process_pool(args.processes) as pool, \
            ParquetBatchWriter(out / "read_features.parquet", READ_FEATURES) as wf, \
            ParquetBatchWriter(out / "read_summary.parquet", READ_SUMMARY) as ws, \
            ParquetBatchWriter(out / "annot_segments.parquet", ANNOT_SEGMENTS) as wa:
        for feats, summ, segs in bounded_imap(pool, annotate_batch, batched(tasks, BATCH), 4 * args.processes):
            wf.write(feats)
            ws.write(summ)
            wa.write(segs)
            n += len(summ["zm"])
            if n % 100_000 < BATCH:
                print(f"{sample.sample}: {n:,} reads annotated, {time.time() - t0:,.0f} s", file=sys.stderr)
    done = {"sample": sample.sample, "fraction": args.fraction, "hifi": hifi.n, "fail": fail.n, "fail_bam_records": fail.n,
            "fail_sub_modulo": modulo, "non_passing": nonp.n, "annotated": n, "seconds": round(time.time() - t0, 1)}
    (out / "annotate.done.json").write_text(json.dumps(done, indent=1))
    print(json.dumps(done), file=sys.stderr)


if __name__ == "__main__":
    main()
