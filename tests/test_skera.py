import json
from pathlib import Path

import polars as pl
import pytest

from kinnex_qc import skera
from synth import complete_with_false_cut, double_mirror, frames, tso_tso_mirror

GOLD = Path(__file__).parent / "golden" / "v10_inputs"


def test_false_cut_found_and_real_adapter_inside():
    f, m, s = frames(complete_with_false_cut())
    fc = skera.false_cuts(s, f)
    assert fc.select("zm", "cut_start", "cut_end", "dr_before").rows() == [(1, 1500, 1514, 3)]
    inside = skera.adapters_inside_sreads(s, f)
    assert inside.select("zm", "di", "name").rows() == [(1, 0, "D")]


def test_cut_at_real_adapter_is_not_false():
    f, m, s = frames(complete_with_false_cut())
    s = s.with_columns(pl.when(pl.col("di") == 0).then(1017).otherwise(pl.col("qe")).alias("qe"),
                       pl.when(pl.col("di") == 1).then(1034).otherwise(pl.col("qs")).alias("qs"))
    assert skera.false_cuts(s, f).height == 0


def test_molecule_coverage_classes():
    f, m, s = frames(complete_with_false_cut())
    cov = skera.molecule_coverage(m, s).sort("start")
    assert cov["cls"].to_list() == ["FUSED", "PARTIAL"]   # S-read 0 covers C-D fully and 47% of D-E; S-read 1 covers 52% of D-E


def test_mirror_one_arm_written():
    f, m, s = frames(tso_tso_mirror())
    cov = skera.molecule_coverage(m, s).filter(pl.col("end_class") == "proper").sort("start")   # pass all molecules of a read
    assert cov["cls"].to_list() == ["NOT_WRITTEN", "CLEAN"]
    st = skera.single_tso_tso_mirrors(m)
    assert st.select("zm", "letter").rows() == [(2, "H")]


def test_best_cover_is_looser_than_clean():
    f, m, s = frames(complete_with_false_cut())
    bc = skera.best_cover(m, s).sort("start")
    assert bc["cover"][0] == 1.0          # "written" under the downstream test's rule although the S-read is a chimera


def test_double_mirror():
    f, m, s = frames(double_mirror())
    assert skera.double_mirrors(m).to_list() == [3]
    assert skera.single_tso_tso_mirrors(m).height == 0


@pytest.mark.real
def test_false_cut_rate_kin_matches_v10(local):
    KIN = local["hash_dir"] / "BD144_kin"
    rs = pl.read_parquet(KIN / "read_summary.parquet", columns=["zm", "source", "skera_passing", "primary_pattern"])
    target = rs.filter((pl.col("source") == "hifi") & pl.col("skera_passing") & (pl.col("primary_pattern") == "complete"))["zm"]
    f = pl.read_parquet(KIN / "read_features.parquet", columns=["zm", "source", "kind", "name", "strand", "start", "end", "edits"]) \
        .filter((pl.col("source") == "hifi") & pl.col("zm").is_in(target))
    s = pl.read_parquet(KIN / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"]).filter(pl.col("zm").is_in(target))
    fc = skera.false_cuts(s, f)
    gold = json.loads((GOLD / "false_cut_anatomy" / "BD144_kin_false_cuts.json").read_text())
    assert fc.height == gold["false_cuts"]
    assert fc["zm"].n_unique() / target.len() == pytest.approx(gold["reads_with_false_cut"], abs=0)
