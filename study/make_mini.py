#!/usr/bin/env python3
"""Build the mini dataset the study's real tests run on: a small slice of each sample's plain-file inputs.

For each sample: the first 1,500 HiFi reads; the segmented and non-passing records of those ZMWs (both files follow
the HiFi ZMW order, so each is read up to the first ZMW past the set); the first 1,000 fail reads; the zmw_metrics
header, its first 20,000 rows and the rows of the kept HiFi and fail ZMWs, in file order; the fltnc aligned records
of the kept HiFi ZMWs; and the whole-cell skera report, copied as it is. BAM records and headers are copied as they
are.

Writes <mini>/<sample>/ with the file names prepare_inputs.sh writes, plus <sample>.fltnc.aligned.bam, and
<mini>/samples_mini.tsv in the samples.tsv format (rows of samples built earlier and not rebuilt now are kept).
<mini> is $KINNEX_QC_MINI, else mini_dir in the TOML file named by $KINNEX_QC_LOCAL (default local.toml).

Usage, from the repository root: study/make_mini.py [--samples PATH] [SAMPLE ...]
"""
import argparse
import gzip
import os
import shutil
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pysam

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))     # study/pipeline/config.py

from config import EXPECTED_COLUMNS, Sample, load_config, load_samples  # noqa: E402

HIFI_READS = 1_500
FAIL_READS = 1_000
ZMW_METRICS_ROWS = 20_000      # data rows kept from the top of zmw_metrics, besides the rows of the kept ZMWs
ZMW_ID = 5                     # index of the zmw_id column in zmw_metrics
THREADS = 4


def first_n(src: Path, dest: Path, n: int) -> list[int]:
    """Copy the first n records of src to dest; return their ZMW ids in file order."""
    zms = []
    with pysam.AlignmentFile(str(src), "rb", check_sq=False, threads=THREADS) as bam, \
            pysam.AlignmentFile(str(dest), "wb", template=bam, threads=THREADS) as out:
        for r in bam:
            if len(zms) == n:
                break
            out.write(r)
            zms.append(r.get_tag("zm"))
    return zms


def same_zmws(src: Path, dest: Path, zms: set[int]) -> int:
    """Copy the records of src whose ZMW is in zms, stopping at the first record outside zms once one inside has been
    seen (records before the first ZMW of the set are skipped). Returns the number of records copied."""
    n, seen = 0, False
    with pysam.AlignmentFile(str(src), "rb", check_sq=False, threads=THREADS) as bam, \
            pysam.AlignmentFile(str(dest), "wb", template=bam, threads=THREADS) as out:
        for r in bam:
            if r.get_tag("zm") in zms:
                seen = True
                out.write(r)
                n += 1
            elif seen:
                break
    return n


def zmw_metrics(src: Path, dest: Path, zms: set[int]) -> int:
    """Copy the header, the first ZMW_METRICS_ROWS rows and every row whose zmw_id is in zms, in file order.
    Returns the number of data rows copied."""
    keep = {str(z).encode() for z in zms}
    n = 0
    with gzip.open(src, "rb") as lines, gzip.open(dest, "wb") as out:
        for i, line in enumerate(lines):
            line = line.rstrip(b"\n")
            if i <= ZMW_METRICS_ROWS or (len(fields := line.split(b",")) > ZMW_ID and fields[ZMW_ID] in keep):
                out.write(line + b"\n")
                n += i > 0
    return n


def fltnc(src: Path, dest: Path, zms: set[int]) -> int:
    """Copy every record of the aligned BAM src whose ZMW is in zms. Returns the number of records copied."""
    n = 0
    with pysam.AlignmentFile(str(src), "rb", threads=THREADS) as bam, \
            pysam.AlignmentFile(str(dest), "wb", template=bam, threads=THREADS) as out:
        for r in bam.fetch(until_eof=True):
            if r.has_tag("zm") and r.get_tag("zm") in zms:
                out.write(r)
                n += 1
    return n


def mini_sample(sample: Sample, mini: Path) -> Sample:
    d = mini / sample.sample
    return Sample(sample=sample.sample, cell=sample.cell, depletion=sample.depletion,
                  hifi_bam=d / "hifi_reads.bam", fail_bam=d / "fail_reads.bam", segmented_bam=d / "segmented.bam",
                  non_passing_bam=d / "segmented.non_passing.bam", zmw_metrics=d / "zmw_metrics.csv.gz",
                  skera_report=d / "read_segmentation.report.json", fltnc_bam=d / f"{sample.sample}.fltnc.aligned.bam")


def build(sample: Sample, mini: Path) -> str:
    m = mini_sample(sample, mini)
    m.hifi_bam.parent.mkdir(parents=True, exist_ok=True)
    hifi = first_n(sample.hifi_bam, m.hifi_bam, HIFI_READS)
    kept = set(hifi)
    n_seg = same_zmws(sample.segmented_bam, m.segmented_bam, kept)
    if not hifi or not n_seg:
        raise RuntimeError(f"{sample.sample}: no HiFi or no segmented records written")
    n_np = same_zmws(sample.non_passing_bam, m.non_passing_bam, kept)
    fail = first_n(sample.fail_bam, m.fail_bam, FAIL_READS)
    n_zmw = zmw_metrics(sample.zmw_metrics, m.zmw_metrics, kept | set(fail))
    shutil.copyfile(sample.skera_report, m.skera_report)
    n_fl = fltnc(sample.fltnc_bam, m.fltnc_bam, kept)
    return (f"{sample.sample}: {len(hifi):,} HiFi reads, {n_seg:,} S-reads, {n_np:,} non-passing, {len(fail):,} fail reads, "
            f"{n_zmw:,} zmw_metrics rows, {n_fl:,} fltnc records")


def mini_dir() -> Path:
    if os.environ.get("KINNEX_QC_MINI"):
        return Path(os.environ["KINNEX_QC_MINI"])
    local = Path(os.environ.get("KINNEX_QC_LOCAL", "local.toml"))
    if local.exists() and "mini_dir" in (values := tomllib.loads(local.read_text())):
        return Path(values["mini_dir"])
    raise SystemExit("make_mini.py: set KINNEX_QC_MINI or mini_dir in local.toml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", help="samples table (default: samples in study/study.toml)")
    ap.add_argument("sample", nargs="*", help="samples to build (default: all)")
    args = ap.parse_args()

    samples = load_samples(args.samples or load_config()["samples"])
    names = args.sample or list(samples)
    unknown = [s for s in names if s not in samples]
    if unknown:
        ap.error(f"not in the samples table: {', '.join(unknown)}")
    mini = mini_dir()
    with ProcessPoolExecutor(len(names)) as pool:
        for line in pool.map(build, [samples[s] for s in names], [mini] * len(names)):
            print(line)

    table = mini / "samples_mini.tsv"
    rows = load_samples(table) if table.exists() else {}
    rows |= {s: mini_sample(samples[s], mini) for s in names}
    rows = {s: rows[s] for s in samples if s in rows} | rows
    table.write_text("\t".join(EXPECTED_COLUMNS) + "\n"
                     + "".join("\t".join(str(getattr(r, c)) for c in EXPECTED_COLUMNS) + "\n" for r in rows.values()))


if __name__ == "__main__":
    main()
