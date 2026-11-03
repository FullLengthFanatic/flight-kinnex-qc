import json
from pathlib import Path

import polars as pl
import pytest

from kinnex_qc import metrics
from synth import complete_with_false_cut, double_mirror, frames, tso_tso_mirror

GOLD = Path(__file__).parent / "golden"


def _read_summary() -> pl.DataFrame:
    """Hand-built read_summary for the three synth reads: zm 1 complete (n_mas 3, passing), zm 2 mirror
    (n_mas 4, not passing), zm 3 mirror (n_mas 5, passing; a double mirror is a "mirror" read)."""
    rows = [(1, "complete", 3, True, 12000), (2, "mirror", 4, False, 6000), (3, "mirror", 5, True, 2500)]
    return pl.DataFrame(rows, schema={"zm": pl.UInt32, "primary_pattern": pl.Utf8, "n_mas": pl.Int16,
                                      "skera_passing": pl.Boolean, "read_len": pl.Int32}, orient="row")


def _molecules() -> pl.DataFrame:
    """Molecules of the same three synth reads: 2 (read 1) + 3 (read 2) + 4 (read 3) = 9 rows."""
    _, mols, _ = frames({**complete_with_false_cut(), **tso_tso_mirror(), **double_mirror()})
    return mols


def test_pattern_shares_complete_and_mirror():
    shares = metrics.pattern_shares(_read_summary())
    assert shares.rows() == [("complete", 1, 1 / 3), ("mirror", 2, 2 / 3)]


def test_pattern_shares_folds_other_patterns():
    rs = pl.DataFrame([(1, "complete"), (2, "skipped_slot"), (3, "repeated"), (4, "unresolved")],
                      schema={"zm": pl.UInt32, "primary_pattern": pl.Utf8}, orient="row")
    shares = metrics.pattern_shares(rs)
    assert shares.rows() == [("complete", 1, 0.25), ("other", 3, 0.75)]


def test_artifact_rates_eight_keys():
    rates = metrics.artifact_rates(_molecules())
    assert rates == {
        "proper_per_1000": pytest.approx(1000 * 6 / 9), "tso_tso_per_1000": pytest.approx(1000 * 2 / 9),
        "r1_r1_per_1000": pytest.approx(1000 * 1 / 9), "tso_only_per_1000": 0.0, "r1_only_per_1000": 0.0,
        "none_per_1000": 0.0, "other_per_1000": 0.0, "internal_per_1000": 0.0}


def test_read_fractions_complete_and_mirror():
    assert metrics.read_fractions(_read_summary()) == {"complete_fraction": pytest.approx(1 / 3), "mirror_fraction": pytest.approx(2 / 3)}


def test_mirror_junction_shares():
    assert metrics.mirror_junction_shares(_molecules()) == {
        "mirror_junctions": 3, "mirror_junction_tso_tso_share": pytest.approx(2 / 3),
        "mirror_junction_r1_r1_share": pytest.approx(1 / 3)}


def test_junction_table_rows():
    jt = metrics.junction_table(_molecules())
    assert jt.rows() == [("B", "R1-R1", 1), ("B", "TSO-TSO", 1), ("H", "TSO-TSO", 1)]


def test_junction_composition():
    """Every row, in order: reads descending, then the label, which breaks the tie of one read each here."""
    jc = metrics.junction_composition(_molecules(), _read_summary())
    assert jc.columns == ["junction_molecules", "reads", "median_molecules", "skera_pass_share"]
    assert jc.rows() == [("TSO-TSO and R1-R1", 1, 4.0, 1.0), ("TSO-TSO only", 1, 3.0, 0.0)]


def test_mirror_size_rule():
    msr = metrics.mirror_size_rule(_molecules(), _read_summary())
    assert msr.rows(named=True) == [{"letter": "H", "reads": 1, "predicted": 3, "median_molecules": 3.0, "share_matching": 1.0}]


def _assert_dict_equal(got: dict, gold: dict, keys):
    for k in keys:
        assert got[k] == gold[k], f"{k}: {got[k]} != {gold[k]}"


def _assert_table_equal(got: pl.DataFrame, gold: pl.DataFrame):
    """got equals v10's table column for column and row for row, every value cast to the golden column's dtype.
    Neither side is re-sorted: each v10 file below lists its rows in the order the function sorts them
    (pattern_shares: pattern; junction_table: letter, molecule; junction_composition: reads descending, then label;
    mirror_size_rule: letter), so the row order is checked too."""
    assert got.columns == gold.columns, f"columns: {got.columns} != {gold.columns}"
    assert got.height == gold.height, f"row count: {got.height} != {gold.height}"
    for col in gold.columns:
        a, b = got[col].cast(gold[col].dtype).to_list(), gold[col].to_list()
        for i, (x, y) in enumerate(zip(a, b)):
            assert x == y, f"{col}[{i}]: {x} != {y}"


@pytest.fixture(scope="module")
def nd_frames(local):
    ND = local["hash_dir"] / "BD144_ND"
    rs = pl.read_parquet(ND / "read_summary.parquet").filter(pl.col("source") == "hifi")
    mols = (pl.scan_parquet(ND / "annot_segments.parquet")
            .filter(pl.col("source") == "hifi")
            .select("zm", "start", "end", "end_class", "internal", "left_token", "right_token").collect())
    return rs, mols


@pytest.mark.real
def test_artifact_rates_match_v10(nd_frames):
    _, mols = nd_frames
    gold_q2 = json.loads((GOLD / "v10_inputs" / "hash_0.18" / "BD144_ND" / "q2" / "q2_summary.json").read_text())
    rates = metrics.artifact_rates(mols)
    _assert_dict_equal(rates, gold_q2, rates.keys())


@pytest.mark.real
def test_read_fractions_match_v10(nd_frames):
    rs, _ = nd_frames
    gold_q2 = json.loads((GOLD / "v10_inputs" / "hash_0.18" / "BD144_ND" / "q2" / "q2_summary.json").read_text())
    _assert_dict_equal(metrics.read_fractions(rs), gold_q2, ["complete_fraction", "mirror_fraction"])


@pytest.mark.real
def test_mirror_junction_shares_match_v10(nd_frames):
    _, mols = nd_frames
    gold_q2 = json.loads((GOLD / "v10_inputs" / "hash_0.18" / "BD144_ND" / "q2" / "q2_summary.json").read_text())
    _assert_dict_equal(metrics.mirror_junction_shares(mols), gold_q2,
                       ["mirror_junctions", "mirror_junction_tso_tso_share", "mirror_junction_r1_r1_share"])


@pytest.mark.real
def test_pattern_shares_match_v10(nd_frames):
    rs, _ = nd_frames
    gold = (pl.read_csv(GOLD / "v10_tables" / "fig2a_chain_patterns.tsv", separator="\t")
            .filter(pl.col("library") == "ND").select("pattern", "reads", "share"))
    _assert_table_equal(metrics.pattern_shares(rs), gold)


@pytest.mark.real
def test_junction_table_matches_v10(nd_frames):
    _, mols = nd_frames
    gold = pl.read_csv(GOLD / "v10_tables" / "fig4b_mirror_junction_letters_ND.tsv", separator="\t")
    _assert_table_equal(metrics.junction_table(mols), gold)


@pytest.mark.real
def test_junction_composition_matches_v10(nd_frames):
    rs, mols = nd_frames
    gold = pl.read_csv(GOLD / "v10_tables" / "S5_mirror_junction_composition_ND.tsv", separator="\t")
    _assert_table_equal(metrics.junction_composition(mols, rs), gold)


@pytest.mark.real
def test_mirror_size_rule_matches_v10(nd_frames):
    rs, mols = nd_frames
    gold = pl.read_csv(GOLD / "v10_tables" / "fig4c_mirror_array_size_rule_ND.tsv", separator="\t")
    _assert_table_equal(metrics.mirror_size_rule(mols, rs), gold)
