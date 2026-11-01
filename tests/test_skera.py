import json
from pathlib import Path

import polars as pl
import pytest

from kinnex_qc import skera
from kinnex_qc.patterns import MAS_ORDER
from synth import complete_with_false_cut, double_mirror, frames, tso_tso_mirror

GOLD = Path(__file__).parent / "golden" / "v10_inputs"


def _sreads(rows, schema={"zm": pl.UInt32, "di": pl.Int16, "qs": pl.Int32, "qe": pl.Int32, "dl": pl.Int8, "dr": pl.Int8}):
    return pl.DataFrame(rows, schema=schema, orient="row")


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


def test_passing_from_labels_forward_and_reversed():
    s = _sreads([
        (10, 0, 0, 17, 0, 1), (10, 1, 17, 34, 1, 2), (10, 2, 34, 51, 2, 3), (10, 3, 51, 68, 3, 4),
        (10, 4, 68, 85, 4, 5), (10, 5, 85, 102, 5, 6), (10, 6, 102, 119, 6, 7), (10, 7, 119, 136, 7, 8),
        (11, 0, 0, 17, 8, 7), (11, 1, 17, 34, 7, 6), (11, 2, 34, 51, 6, 5), (11, 3, 51, 68, 5, 4),
        (11, 4, 68, 85, 4, 3), (11, 5, 85, 102, 3, 2), (11, 6, 102, 119, 2, 1), (11, 7, 119, 136, 1, 0),
        (12, 0, 0, 17, 0, 1), (12, 1, 17, 34, 1, 2), (12, 2, 34, 51, 2, 4),
    ])
    out = dict(skera.passing_from_labels(s, order_len=9).select("zm", "passing").rows())
    assert out == {10: True, 11: True, 12: False}


def test_passing_from_labels_mirror_arm_alone_is_not_passing():
    f, m, s = frames(tso_tso_mirror())
    out = dict(skera.passing_from_labels(s, order_len=9).select("zm", "passing").rows())
    assert out == {2: False}


def test_skipped_adapters_matches_false_cut():
    f, m, s = frames(complete_with_false_cut())
    fc = skera.false_cuts(s, f)
    sk = skera.skipped_adapters(fc, s, f)
    assert sk.select("zm", "cut_start", "missed", "in_sread", "dist").rows() == [(1, 1500, "D", 1, 490)]


def test_false_cut_stats_uses_the_passed_in_passing_and_fc():
    """Fix round 2, finding 1: false_cut_stats must take its counts from the passing/fc frames it is
    given, not recompute passing_from_labels/false_cuts itself (the runner now computes both once and
    passes them in). A caller-supplied fc with the one real false cut duplicated must show up as 2
    false cuts: if false_cut_stats instead recomputed false_cuts(s, f) internally and ignored the fc
    argument, this would still read 1."""
    f, m, s = frames(complete_with_false_cut())
    read_summary = pl.DataFrame({"zm": [1], "primary_pattern": ["complete"]}, schema={"zm": pl.UInt32, "primary_pattern": pl.Utf8})
    # passing is given directly (zm 1's two partial S-reads would not derive True on their own; the
    # point here is only that false_cut_stats must use the frame it is given, not re-derive it).
    passing = pl.DataFrame({"zm": [1], "passing": [True]}, schema={"zm": pl.UInt32, "passing": pl.Boolean})
    fc = skera.false_cuts(s, f)
    stats = skera.false_cut_stats(read_summary, f, s, passing=passing, fc=fc, order=MAS_ORDER)
    assert stats["false_cuts"] == 1
    stats_dup = skera.false_cut_stats(read_summary, f, s, passing=passing, fc=pl.concat([fc, fc]), order=MAS_ORDER)
    assert stats_dup["false_cuts"] == 2


def test_mirror_arm_classes_regions():
    f, m, s = frames(tso_tso_mirror())
    arm = skera.mirror_arm_classes(m, s).sort("region")
    assert arm.select("zm", "region", "cls").rows() == [(2, "after", "CLEAN"), (2, "before", "NOT_WRITTEN")]


def test_mirror_arm_stats_shares():
    f, m, s = frames(tso_tso_mirror())
    stats = skera.mirror_arm_stats(m, s)
    assert stats["after_clean_share"] == 1.0
    assert stats["before_clean_share"] == 0.0
    assert stats["after_counts"] == {"CLEAN": 1, "FUSED": 0, "PARTIAL": 0, "NOT_WRITTEN": 0}
    assert stats["before_counts"] == {"CLEAN": 0, "FUSED": 0, "PARTIAL": 0, "NOT_WRITTEN": 1}


def test_budget_stats_by_hand():
    reads = {**complete_with_false_cut(), **tso_tso_mirror(), **double_mirror()}
    f, m, s = frames(reads)
    read_summary = pl.DataFrame({"zm": [1, 2, 3]}, schema={"zm": pl.UInt32})
    cls = skera.budget_classes(m, s)
    stats = skera.budget_stats(read_summary, cls, order_len=9)
    assert stats["molecules"] == pytest.approx(3.0)
    assert stats["unfilled_slots"] == pytest.approx(5.0)
    assert stats["artifact_tso_tso"] == pytest.approx(2 / 3)
    assert stats["artifact_r1_r1"] == pytest.approx(1 / 3)
    assert stats["artifact_one_primer_or_none"] == pytest.approx(0.0)
    assert stats["artifact_fused"] == pytest.approx(0.0)
    assert stats["artifact"] == pytest.approx(1.0)
    assert stats["usable"] == pytest.approx(2.0)
    assert stats["usable_clean"] == pytest.approx(1 / 3)
    assert stats["usable_not_clean"] == pytest.approx(5 / 3)


@pytest.mark.real
def test_false_cut_letters_and_skipped_adapter_letters_match_v10(local):
    KIN = local["hash_dir"] / "BD144_kin"
    rs = pl.read_parquet(KIN / "read_summary.parquet", columns=["zm", "source", "primary_pattern"]).filter(pl.col("source") == "hifi")
    complete_zm = rs.filter(pl.col("primary_pattern") == "complete")["zm"]
    f = pl.read_parquet(KIN / "read_features.parquet", columns=["zm", "source", "kind", "name", "strand", "start", "end", "edits"]) \
        .filter((pl.col("source") == "hifi") & pl.col("zm").is_in(complete_zm))
    s = pl.read_parquet(KIN / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"]).filter(pl.col("zm").is_in(complete_zm))
    rs_complete = rs.filter(pl.col("primary_pattern") == "complete").select("zm", "primary_pattern")
    passing = skera.passing_from_labels(s, order_len=len(MAS_ORDER))
    fc = skera.false_cuts(s, f)
    stats = skera.false_cut_stats(rs_complete, f, s, passing=passing, fc=fc, order=MAS_ORDER)
    gold = json.loads((GOLD / "false_cut_anatomy" / "BD144_kin_false_cuts.json").read_text())
    assert stats["complete_passing_reads"] == 1_950_193
    assert stats["false_cuts"] == gold["false_cuts"]
    assert stats["false_cut_reads_share"] == pytest.approx(gold["reads_with_false_cut"], abs=0)
    classes = pl.read_csv(GOLD / "false_cut_anatomy" / "BD144_kin_false_cut_classes.tsv", separator="\t")
    skera_letter = dict(classes.filter(pl.col("field") == "skera_letter").select("value", "false_cuts").rows())
    missed = {(row[0] if row[0] else "none"): row[1]
              for row in classes.filter(pl.col("field") == "missed").select("value", "false_cuts").rows()}
    assert stats["false_cut_letters"] == skera_letter
    assert stats["skipped_adapter_letters"] == missed


@pytest.mark.real
def test_mirror_arms_match_v10_nd(local):
    ND = local["hash_dir"] / "BD144_ND"
    m = pl.read_parquet(ND / "annot_segments.parquet",
                         columns=["zm", "source", "start", "end", "end_class", "internal", "left_token", "right_token"]) \
        .filter(pl.col("source") == "hifi").drop("source")
    s = pl.read_parquet(ND / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"])
    stats = skera.mirror_arm_stats(m, s)
    assert stats["after_clean_share"] == pytest.approx(0.8588701530810475, abs=0)
    assert stats["before_clean_share"] == pytest.approx(0.08099823120678912, abs=0)
    gold = pl.read_csv(GOLD.parent / "v10_tables" / "S10_mirror_molecules_written_by_skera_ND.tsv", separator="\t") \
        .filter(pl.col("set") == "nd_hash_skera26")
    for region in ("after", "before"):
        want = dict(gold.filter(pl.col("region") == region).select("class", "n").rows())
        assert stats[f"{region}_counts"] == want


@pytest.mark.real
def test_budget_matches_v10_nd(local):
    ND = local["hash_dir"] / "BD144_ND"
    rs = pl.read_parquet(ND / "read_summary.parquet", columns=["zm", "source"]).filter(pl.col("source") == "hifi").select("zm")
    m = pl.read_parquet(ND / "annot_segments.parquet", columns=["zm", "source", "start", "end", "end_class", "internal"]) \
        .filter(pl.col("source") == "hifi").drop("source")
    s = pl.read_parquet(ND / "segments.parquet", columns=["zm", "di", "qs", "qe", "dl", "dr"])
    cls = skera.budget_classes(m, s)
    stats = skera.budget_stats(rs, cls, order_len=9)
    gold = pl.read_csv(GOLD.parent / "v10_tables" / "fig6_molecule_budget.tsv", separator="\t").filter(pl.col("library") == "ND")
    row = gold.to_dicts()[0]
    for key, gold_key in (("molecules", "molecules"), ("unfilled_slots", "unfilled_slots"),
                          ("artifact_tso_tso", "artifact_tso_tso"), ("artifact_r1_r1", "artifact_r1_r1"),
                          ("artifact_one_primer_or_none", "artifact_one_primer_or_none"),
                          ("artifact_fused", "artifact_fused"), ("usable", "usable"),
                          ("usable_not_clean", "not_written_cleanly")):
        assert stats[key] == pytest.approx(row[gold_key], abs=0), key


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
