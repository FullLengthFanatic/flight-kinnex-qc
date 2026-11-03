#!/usr/bin/env python3
"""Stage 01: classify every skera S-read of one sample into segments.parquet.

Usage, from the repository root:
  study/pipeline/s01_extract_segments.py --sample <name> [--samples PATH] [--out-root DIR] [--fraction F] [--processes N]
Defaults come from study/study.toml. Below fraction 1 only the S-reads of the ZMWs in the hash sample are read.
"""
import argparse
import json
import sys
import time

from config import load_config, load_samples, out_dir
from kinnex_qc.classify import classify_segment, tso_toward_lower
from kinnex_qc.io import iter_bam
from kinnex_qc.parallel import ParquetBatchWriter, batched, bounded_imap, process_pool
from kinnex_qc.sampling import in_hash_sample
from kinnex_qc.scan import encode_hits
from kinnex_qc.schemas import SEGMENTS

BATCH = 20_000


def classify_batch(batch: list[tuple]) -> dict[str, list]:
    cols = {f.name: [] for f in SEGMENTS}
    for zm, di, dl, dr, qs, qe, np_, rq, seq in batch:
        call = classify_segment(seq)
        row = {
            "zm": zm, "di": di, "dl": dl, "dr": dr,
            "pos": min(dl, dr), "direction": 1 if dl < dr else -1,
            "qs": qs, "qe": qe, "sread_len": qe - qs, "np": np_, "rq": rq,
            "end_class": call.end_class, "tso_side": call.tso_side,
            "tso_toward_lower": tso_toward_lower(call, dl, dr),
            "internal": call.internal, "n_internal": len(call.internal_hits),
            "internal_hits": encode_hits(list(call.internal_hits)),
            "head_hits": encode_hits(list(call.head_hits)),
            "tail_hits": encode_hits(list(call.tail_hits)),
            "bead_ok": call.bead_ok, "bead_shift": call.bead_shift,
            "polyT_len": call.polyT_len, "polyT_anchored": call.polyT_anchored,
            "insert_len": call.insert_len,
        }
        for k, v in row.items():
            cols[k].append(v)
    return cols


def records(sample, fraction: float):
    keep = (lambda zm: in_hash_sample(zm, fraction)) if fraction < 1 else None
    for r in iter_bam(sample.segmented_bam, threads=4, keep=keep):
        yield (r.get_tag("zm"), r.get_tag("di"), r.get_tag("dl"), r.get_tag("dr"),
               r.get_tag("qs"), r.get_tag("qe"), r.get_tag("np"), r.get_tag("rq"), r.query_sequence)


def main() -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--samples", default=config["samples"], help="samples table")
    ap.add_argument("--out-root", default=config["out_root"], help="output root")
    ap.add_argument("--fraction", type=float, default=config["fraction"], help="hash sample of the cell's ZMWs; 1 reads them all")
    ap.add_argument("--processes", type=int, default=config["processes"])
    args = ap.parse_args()

    sample = load_samples(args.samples)[args.sample]
    out = out_dir(args.out_root, sample.sample, args.fraction)
    t0 = time.time()
    n = 0
    with process_pool(args.processes) as pool, ParquetBatchWriter(out / "segments.parquet", SEGMENTS) as writer:
        for cols in bounded_imap(pool, classify_batch, batched(records(sample, args.fraction), BATCH), 4 * args.processes):
            writer.write(cols)
            n += len(cols["zm"])
            if n % 1_000_000 < BATCH:
                print(f"{sample.sample}: {n:,} S-reads, {n / (time.time() - t0):,.0f}/s", file=sys.stderr)
    done = {"sample": sample.sample, "fraction": args.fraction, "s_reads": n, "seconds": round(time.time() - t0, 1)}
    (out / "segments.done.json").write_text(json.dumps(done, indent=1))
    print(json.dumps(done), file=sys.stderr)


if __name__ == "__main__":
    main()
