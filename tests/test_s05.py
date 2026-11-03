"""Tests for study/pipeline/s05_fate.py (join logic), and stages 04 and 05 on the mini dataset."""
import polars as pl
import pytest

from kinnex_qc.schemas import FATE
from pipeline_helpers import run_stage

import s05_fate as fate_mod

U = {"zm": pl.UInt32, "di": pl.Int16}


def keys(pairs, extra=None):
    d = {"zm": [p[0] for p in pairs], "di": [p[1] for p in pairs]}
    if extra:
        d.update(extra)
    return pl.DataFrame(d, schema_overrides=U)


def test_build_fate():
    seg = pl.DataFrame({"zm": [1, 1, 2], "di": [0, 1, 0], "pos": [0, 1, 0], "end_class": ["proper", "tso_tso", "proper"],
                        "sread_len": [1000, 200, 900]}, schema_overrides={**U, "pos": pl.Int8})
    nonp = pl.DataFrame({"zm": [2]}, schema={"zm": pl.UInt32})
    pairs = keys([(1, 0), (1, 1)], {"lima_pair": ["5p--3p", "5p--5p"]})
    refine = {f"refine_polyA_{n}": keys([]) for n in (5, 8, 12, 20)}
    refine["refine_asrun"] = keys([(1, 0)])
    f = fate_mod.build_fate(seg, nonp, pairs, keys([(1, 0)]), refine, keys([(1, 0)]))
    assert f.columns == [x.name for x in FATE]
    assert f["lima_pair"].to_list() == ["5p--3p", "5p--5p", "none"]
    assert f["skera_passing"].to_list() == [True, True, False]
    assert f["mapped_fltnc"].to_list() == [True, False, False]
    assert f["refine_polyA_5"].to_list() == [False] * 3


@pytest.mark.real
def test_lima_refine_and_fate_mini(mini, tmp_path):
    for stage in ("s01_extract_segments.py", "s02_annotate_reads.py", "s04_lima_refine.sh", "s05_fate.py"):
        run_stage(stage, "BD144_ND", tmp_path, mini / "samples_mini.tsv", "--fraction", "1")
    out = tmp_path / "BD144_ND"
    assert (out / "lima" / "BD144_ND.isoseq.FlashBD_5p--FlashBD_3p.bam").exists()
    assert all((out / "refine" / f"BD144_ND.polyA_{n}.bam").exists() for n in (5, 8, 12, 20))
    assert not (tmp_path / "tmp").exists()                  # fraction 1: lima reads the segmented BAM, no temporary copy
    f = pl.read_parquet(out / "fate.parquet")
    seg = pl.read_parquet(out / "segments.parquet")
    assert f.height == seg.height
    assert f["lima_pass"].mean() > 0.9 and f["mapped_fltnc"].mean() > 0.9
    assert f.filter(pl.col("mapped_fltnc"))["lima_pass"].mean() > 0.995    # gate 4 condition
