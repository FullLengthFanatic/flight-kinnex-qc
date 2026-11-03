"""Tests for study/pipeline/s03_build_reads.py."""
import gzip

import polars as pl
import pytest

from kinnex_qc.schemas import ZMW
from pipeline_helpers import PIPELINE  # noqa: F401  (puts study/pipeline on sys.path)

import s03_build_reads as build_mod
from config import Sample


def segs(zm, dls, drs, cls="proper"):
    return [{"zm": zm, "dl": dl, "dr": dr, "pos": min(dl, dr), "direction": 1 if dl < dr else -1,
             "sread_len": 1000, "end_class": cls} for dl, dr in zip(dls, drs)]


def test_build_reads_full_array_rules():
    rows = (segs(1, range(8), range(1, 9))                     # forward full array
            + segs(2, range(8, 0, -1), range(7, -1, -1))       # reverse full array
            + segs(3, range(1, 8), range(2, 9))                # 7 segments, missing pos 0
            + segs(4, range(8), range(1, 9))                   # full, but non-passing
            + segs(5, [0, 1, 2, 3, 4, 5, 6, 8], [1, 2, 3, 4, 5, 6, 7, 7]))   # last segment reversed
    seg = pl.DataFrame(rows, schema_overrides={"zm": pl.UInt32, "dl": pl.Int8, "dr": pl.Int8, "pos": pl.Int8, "direction": pl.Int8})
    hifi = pl.DataFrame({"zm": [1, 2, 3, 4, 5, 6], "read_len": [8200] * 6, "np": [10] * 6, "rq": [0.999] * 6},
                        schema_overrides={"zm": pl.UInt32})
    nonp = pl.DataFrame({"zm": [4]}, schema={"zm": pl.UInt32})
    r = build_mod.build_reads(seg, hifi, nonp).sort("zm")
    assert r["full_array"].to_list() == [True, True, False, False, False, False]
    assert r["passing"].to_list() == [True, True, True, False, True, True]
    assert r["direction"].to_list()[:2] == [1, -1]
    assert r.filter(pl.col("zm") == 6).row(0, named=True)["n_segments"] == 0
    assert r.filter(pl.col("zm") == 6)["unsegmented_bp"].item() == 8200
    assert r.filter(pl.col("zm") == 1)["unsegmented_bp"].item() == 200
    assert r["positions"][0] == "0,1,2,3,4,5,6,7"


def test_read_zmw_metrics_reads_the_plain_file(tmp_path):
    path = tmp_path / "zmw_metrics.csv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("read_length,pol_length,read_qual,hifi,num_passes,zmw_id\n330,330,-1,1,0,5572078\n11498,12000,0.75,0,2,5572055\n")
    sample = Sample(*["x"] * 3, *[tmp_path / "unused"] * 4, path, *[tmp_path / "unused"] * 2)
    zmw = build_mod.read_zmw_metrics(sample)
    assert zmw.columns == [f.name for f in ZMW]
    assert zmw["zm"].to_list() == [5572078, 5572055]
    assert zmw["pol_length"].to_list() == [330, 12000]
    assert zmw.schema["zm"] == pl.UInt32 and zmw.schema["hifi"] == pl.Int8


@pytest.mark.real
def test_reads_and_zmw(nd_out):
    reads = pl.read_parquet(nd_out / "reads.parquet")
    assert reads.height == 1500
    assert 0.5 < reads["full_array"].mean() < 0.95          # ND has fewer full arrays than the depleted samples
    assert (reads.filter(~pl.col("passing"))["full_array"] == False).all()  # noqa: E712
    zmw = pl.read_parquet(nd_out / "zmw.parquet")
    assert zmw.height > 20_000 and zmw["pol_length"].null_count() == 0
