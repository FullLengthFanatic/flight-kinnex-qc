#!/usr/bin/env python3
"""Calibration set: the first 200,000 HiFi reads of the BD144_ND cell, used where the same reads must be compared
between methods (skera 1.2.0 against 26.1.0, the TSO-at-both-ends check, the downstream test, skera's false cuts).
They are not a random sample of the cell.

1. nd_first200k.hifi.nomod.bam: the first 200,000 records of BD144_ND's HiFi BAM, base-modification tags (MM, ML)
   removed (observed: skera 1.2.0 stops with "cannot create std::vector larger than max_size()" on the first 50 of
   these records with the tags, and runs on the same records without them). Made as v10 made it, with the three
   samtools 1.13 commands its @PG lines record:
       samtools view -h -  (the HiFi BAM on stdin; its first 200,000 records kept, as head would)
       samtools view -b -o <out>/nd_first200k.hifi.bam -
       samtools view -@4 -b -x MM -x ML -o nd_first200k.hifi.nomod.bam nd_first200k.hifi.bam  (in <out>)
   The records pass through SAM text on the way. Observed in v10's file: that trip rounds the float tags (ec, rq, sn) to
   6 significant digits and changes the type codes of integer tags, and v10's calibration outputs carry those rounded
   values (rq in nd_first200k_truth/read_summary.parquet, and the rq the downstream test copies into the molecules it
   gives lima and isoseq refine). Written straight from the HiFi BAM, the file would differ from v10's in those
   digits; with the same trip and the same samtools and htslib versions, a rebuild equals v10's file record for record,
   so every calibration number can be reproduced from the cell's HiFi BAM. htslib writes the SAM text, and its version
   matters too (observed: samtools 1.13 with htslib 1.21, bioconda's default for samtools=1.13, writes 5 of the
   200,000 records differently from v10's file, in sn; with htslib 1.13, none). samtools is --samtools (default:
   samtools of study/study.toml, else samtools in the --samtools-env environment, envs/samtools113.yml); a samtools
   other than 1.13 or an htslib other than 1.13 stops the script, and both versions go to stderr, the run log.
   nd_first200k.hifi.bam is removed afterwards (v10's folder does not keep it). Written only if
   nd_first200k.hifi.nomod.bam is absent.
2. nd_first200k_truth/: our full-read annotation of these reads with the study's stage 02 function
   (kinnex_qc.runner.annotate_batch: read_summary, annot_segments, read_features), plus the S-reads skera 26.1.0
   delivered for them (segments.parquet: zm, di, qs, qe, dl, dr) and skera's verdict (skera_passing: the ZMW is not in
   the delivered non-passing BAM). The segmented and non-passing BAMs are read from their first record of these ZMWs
   up to the first record past them: both files follow the HiFi BAM's ZMW order.
skera 1.2.0 (bioconda pbskera=1.2.0) then runs on the BAM of step 1 with the Kinnex 8-fold adapters: skera_split.py.
The BD144_ND row of --samples gives the input BAMs.
Usage, from the repository root:
  study/calibration/build_calibration_set.py --out DIR [--samples PATH] [--processes N] [--samtools PATH] [--samtools-env NAME|PATH]
--samples, --processes and --samtools-env default to samples, processes and samtools_env of study/study.toml and
study.local.toml. --out is required: calibration_dir holds v10's calibration set, the reference this script's output is
compared with, and is never written.
"""
import argparse
import subprocess
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
SAMTOOLS = ("samtools 1.13", "Using htslib 1.13")     # samtools as v10's @PG lines name it, htslib as observed


def same_zmws(records: Iterable[pysam.AlignedSegment], zms: set[int]) -> Iterator[pysam.AlignedSegment]:
    """The records whose ZMW is in zms, from the first such record up to the first record outside zms after it."""
    seen = False
    for r in records:
        if r.get_tag("zm") in zms:
            seen = True
            yield r
        elif seen:
            return


def first_records(samtools: list[str], hifi_bam: Path, here: Path, n: int) -> None:
    """nd_first200k.hifi.nomod.bam in here: the first n records of hifi_bam through v10's three samtools commands."""
    full = here / "nd_first200k.hifi.bam"
    with open(hifi_bam, "rb") as src:
        view = subprocess.Popen(samtools + ["view", "-h", "-"], stdin=src, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        write = subprocess.Popen(samtools + ["view", "-b", "-o", str(full), "-"], stdin=subprocess.PIPE)
        records = 0
        for line in view.stdout:
            if not line.startswith(b"@"):          # a SAM record line (a read name never starts with @)
                if records == n:
                    break
                records += 1
            write.stdin.write(line)
        write.stdin.close()
        view.stdout.close()                         # as with head: view stops on the closed pipe
        err = view.stderr.read()
        if view.wait() and records < n:
            raise subprocess.CalledProcessError(view.returncode, view.args, stderr=err)
    if write.wait():
        raise subprocess.CalledProcessError(write.returncode, write.args)
    subprocess.run(samtools + ["view", "-@4", "-b", "-x", "MM", "-x", "ML", "-o", "nd_first200k.hifi.nomod.bam", "nd_first200k.hifi.bam"],
                   cwd=here, check=True)
    full.unlink()


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=config["samples"], help="samples table")
    ap.add_argument("--out", required=True, help="calibration folder to write (never calibration_dir, v10's reference set)")
    ap.add_argument("--processes", type=int, default=config["processes"])
    ap.add_argument("--samtools", default=config.get("samtools"),
                    help="samtools 1.13 executable (default: samtools of study/study.toml, else samtools in the --samtools-env environment)")
    ap.add_argument("--samtools-env", default=config["samtools_env"], help="conda environment with samtools 1.13 and htslib 1.13 (a name, or a path)")
    args = ap.parse_args(argv)
    s = load_samples(args.samples)[SAMPLE]
    here = Path(args.out)
    here.mkdir(parents=True, exist_ok=True)
    bam = here / "nd_first200k.hifi.nomod.bam"
    if not bam.exists():
        run = ["conda", "run", "--no-capture-output", "-p" if "/" in args.samtools_env else "-n", args.samtools_env]
        samtools = [args.samtools] if args.samtools else run + ["samtools"]
        version = subprocess.run(samtools + ["--version"], capture_output=True, check=True).stdout.decode(errors="replace").splitlines()[:2]
        if (version[0], version[1].split("+")[0]) != SAMTOOLS:    # a distribution build may say 1.13+ds
            sys.exit(f"build_calibration_set.py: needs samtools 1.13 with htslib 1.13, as v10's calibration BAM; {' '.join(samtools)} is "
                     + ", ".join(version))
        print("\n".join(version), file=sys.stderr)
        first_records(samtools, Path(s.hifi_bam), here, N)
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
