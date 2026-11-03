#!/usr/bin/env python3
"""Stage 03: build reads.parquet (one row per HiFi ZMW) and zmw.parquet.

Needs segments.parquet (stage 01) and the read indexes (stage 02).
Usage, from the repository root:
  study/pipeline/s03_build_reads.py --sample <name> [--samples PATH] [--out-root DIR] [--fraction F] [--processes N]
Defaults come from study/study.toml. zmw.parquet holds every row of the sample's zmw_metrics file, whatever the fraction.
"""
import argparse
import gzip
import io
import json

import polars as pl
import pyarrow.parquet as pq

from config import load_config, load_samples, out_dir
from kinnex_qc.schemas import READS, ZMW


def build_reads(segments: pl.DataFrame, hifi_index: pl.DataFrame, non_passing_index: pl.DataFrame) -> pl.DataFrame:
    per_zmw = segments.group_by("zm").agg(
        pl.len().alias("n_segments"),
        pl.col("direction").mode().first().alias("direction"),
        pl.col("direction").n_unique().alias("n_dir"),
        pl.col("pos").sort().cast(pl.Utf8).str.join(",").alias("positions"),
        ((pl.col("dl").cast(pl.Int16) - pl.col("dr").cast(pl.Int16)).abs() == 1).all().alias("adjacent"),
        pl.col("sread_len").sum().alias("total_sread_len"),
        pl.col("sread_len").max().alias("max_sread_len"),
        (pl.col("end_class") == "proper").sum().alias("n_proper"),
    )
    reads = (
        hifi_index.join(per_zmw, on="zm", how="left")
        .with_columns(
            (~pl.col("zm").is_in(non_passing_index["zm"])).alias("passing"),
            pl.col("n_segments").fill_null(0),
            pl.col("positions").fill_null(""),
            pl.col("total_sread_len").fill_null(0),
            pl.col("max_sread_len").fill_null(0),
            pl.col("n_proper").fill_null(0),
        )
        .with_columns(
            (pl.col("passing") & (pl.col("n_segments") == 8) & (pl.col("positions") == "0,1,2,3,4,5,6,7")
             & pl.col("adjacent").fill_null(False) & (pl.col("n_dir") == 1).fill_null(False)).alias("full_array"),
            (pl.col("read_len") - pl.col("total_sread_len")).alias("unsegmented_bp"),
        )
    )
    return reads.select([f.name for f in READS])


def read_zmw_metrics(sample) -> pl.DataFrame:
    with gzip.open(sample.zmw_metrics, "rb") as fh:
        raw = fh.read()
    dtypes = {"read_length": pl.Int32, "pol_length": pl.Int32, "read_qual": pl.Float32,
              "hifi": pl.Int8, "num_passes": pl.Int32, "zmw_id": pl.UInt32}
    return pl.read_csv(io.BytesIO(raw), schema_overrides=dtypes).rename({"zmw_id": "zm"}).select([f.name for f in ZMW])


def write(df: pl.DataFrame, path, schema) -> None:
    pq.write_table(df.to_arrow().cast(schema), path)


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
    segments = pl.read_parquet(out / "segments.parquet", columns=["zm", "dl", "dr", "pos", "direction", "sread_len", "end_class"])
    reads = build_reads(segments, pl.read_parquet(out / "hifi_index.parquet"), pl.read_parquet(out / "non_passing_index.parquet"))
    write(reads, out / "reads.parquet", READS)
    zmw = read_zmw_metrics(sample)
    write(zmw, out / "zmw.parquet", ZMW)
    done = {"sample": sample.sample, "fraction": args.fraction, "reads": reads.height,
            "full_array": int(reads["full_array"].sum()), "zmw_rows": zmw.height}
    (out / "reads.done.json").write_text(json.dumps(done, indent=1))
    print(json.dumps(done))


if __name__ == "__main__":
    main()
