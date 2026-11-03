#!/usr/bin/env python3
"""Stage 05: fate of every S-read, from skera to the existing mapped fltnc BAM.

Joins segments.parquet with non_passing_index.parquet, the lima and refine outputs of stage 04
and the primary mapped records of the sample's fltnc BAM, all on (zm, di).
Usage, from the repository root:
  study/pipeline/s05_fate.py --sample <name> [--samples PATH] [--out-root DIR] [--fraction F] [--processes N]
Defaults come from study/study.toml.
"""
import argparse
import json

import polars as pl
import pyarrow.parquet as pq
import pysam

from config import load_config, load_samples, out_dir
from kinnex_qc.schemas import FATE

PAIR_NAMES = {(0, 1): "5p--3p", (1, 0): "5p--3p", (0, 0): "5p--5p", (1, 1): "3p--3p"}


def bam_keys(path, primary_mapped_only: bool = False, with_pair: bool = False) -> pl.DataFrame:
    """(zm, di[, lima_pair]) of every record of a BAM."""
    zm, di, pair = [], [], []
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        for r in bam.fetch(until_eof=True):
            if primary_mapped_only and (r.is_unmapped or r.is_secondary or r.is_supplementary):
                continue
            zm.append(r.get_tag("zm"))
            di.append(r.get_tag("di"))
            if with_pair:
                pair.append(PAIR_NAMES[tuple(r.get_tag("bc"))])
    cols = {"zm": pl.Series(zm, dtype=pl.UInt32), "di": pl.Series(di, dtype=pl.Int16)}
    if with_pair:
        cols["lima_pair"] = pl.Series(pair, dtype=pl.Utf8)
    return pl.DataFrame(cols).unique(["zm", "di"])


def flag(frame: pl.DataFrame, keys: pl.DataFrame, name: str) -> pl.DataFrame:
    return frame.join(keys.with_columns(pl.lit(True).alias(name)), on=["zm", "di"], how="left") \
        .with_columns(pl.col(name).fill_null(False))


def build_fate(segments, non_passing, pairs, lima_pass, refine: dict, mapped) -> pl.DataFrame:
    fate = segments.select("zm", "di", "pos", "end_class", "sread_len").with_columns(
        (~pl.col("zm").is_in(non_passing["zm"])).alias("skera_passing"))
    fate = fate.join(pairs, on=["zm", "di"], how="left").with_columns(pl.col("lima_pair").fill_null("none"))
    fate = flag(fate, lima_pass, "lima_pass")
    for name, keys in refine.items():
        fate = flag(fate, keys, name)
    fate = flag(fate, mapped, "mapped_fltnc")
    return fate.select([f.name for f in FATE])


def main() -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--samples", default=config["samples"], help="samples table")
    ap.add_argument("--out-root", default=config["out_root"], help="output root")
    ap.add_argument("--fraction", type=float, default=config["fraction"], help="names the output directory of a hash sample run")
    ap.add_argument("--processes", type=int, default=config["processes"], help="unused; accepted for a uniform CLI")
    args = ap.parse_args()

    sample = load_samples(args.samples)[args.sample]
    out = out_dir(args.out_root, sample.sample, args.fraction)
    s = sample.sample
    refine = {"refine_asrun": bam_keys(out / "refine" / f"{s}.asrun.bam")}
    for n in (5, 8, 12, 20):
        refine[f"refine_polyA_{n}"] = bam_keys(out / "refine" / f"{s}.polyA_{n}.bam")
    fate = build_fate(
        pl.read_parquet(out / "segments.parquet", columns=["zm", "di", "pos", "end_class", "sread_len"]),
        pl.read_parquet(out / "non_passing_index.parquet", columns=["zm"]),
        bam_keys(out / "lima" / f"{s}.pairs.bam", with_pair=True),
        bam_keys(out / "lima" / f"{s}.isoseq.FlashBD_5p--FlashBD_3p.bam"),
        refine,
        bam_keys(sample.fltnc_bam, primary_mapped_only=True),
    )
    pq.write_table(fate.to_arrow().cast(FATE), out / "fate.parquet")
    summary = {c: int(fate[c].sum()) for c in ("skera_passing", "lima_pass", "refine_asrun", "mapped_fltnc")}
    done = {"sample": s, "fraction": args.fraction, "s_reads": fate.height, **summary,
            "lima_pair": dict(fate["lima_pair"].value_counts().iter_rows())}
    (out / "fate.done.json").write_text(json.dumps(done, indent=1))
    print(json.dumps(done))


if __name__ == "__main__":
    main()
