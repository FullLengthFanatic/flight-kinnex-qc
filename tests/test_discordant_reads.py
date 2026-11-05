"""Tests for study/analyses/discordant_reads.py: the reads where skera's cuts and our adapters disagree, on hand-built
reads with known answers, the review page of the drawn reads, and the tables on the v10 hash sample against v10's."""
import json
import re

import polars as pl
import pytest

from pipeline_helpers import GOLDEN_INPUTS, golden_diffs, run_analysis
from synth import complete_with_false_cut, frames

import discordant_reads as dr


def _reads() -> dict:
    """zm 1: complete_with_false_cut, a cut without our adapter (1500-1514) and our D inside the first S-read. zm 2: S-reads
    cut at our adapters. zm 3: no adapter of ours, two S-reads, its first starting at 5. zm 4: our D' (1 edit) inside its
    one S-read. zm 5: not passing, a cut without an adapter."""
    reads = complete_with_false_cut()
    reads[2] = {"adapters": [("C", "+", 0, 17, 0), ("D", "+", 1017, 1034, 0), ("E", "+", 2034, 2051, 0)], "molecules": [],
                "sreads": [(17, 1017, 2, 3), (1034, 2034, 3, 4)]}
    reads[3] = {"adapters": [], "molecules": [], "sreads": [(5, 900, 0, 1), (920, 1800, 1, 2)]}
    reads[4] = {"adapters": [("C", "-", 0, 17, 0), ("D", "-", 1017, 1034, 1), ("E", "-", 2034, 2051, 0)], "molecules": [],
                "sreads": [(17, 2034, 4, 2)]}
    reads[5] = {"adapters": [], "molecules": [], "sreads": [(17, 900, 0, 1), (920, 1800, 1, 2)]}
    return reads


TABLES = ["discordant_all_in_pool.tsv", "discordant_reads_auto.tsv", "discordant_reads_zmws.tsv", "discordant_summary.json"]


def _sample_dir(path):
    feats, mols, sreads = frames(_reads())
    path.mkdir()
    feats.with_columns(pl.lit("hifi").alias("source")).write_parquet(path / "read_features.parquet")
    sreads.write_parquet(path / "segments.parquet")
    (mols.with_columns(pl.lit("hifi").alias("source"), pl.lit(None, pl.Int8).alias("pos"), pl.lit(None, pl.Int16).alias("polyT_len"),
                       pl.lit(None, pl.Utf8).alias("tso_side"))
     .write_parquet(path / "annot_segments.parquet"))
    pl.DataFrame({"zm": [1, 2, 3, 4, 5], "source": "hifi", "read_len": [2051, 2051, 1800, 2051, 1800],
                  "chain": ["C D E", "C D E", "", "E' D' C'", ""], "primary_pattern": ["complete", "complete", "unresolved", "complete", "unresolved"],
                  "skera_passing": [True, True, True, True, False]},
                 schema_overrides={"zm": pl.UInt32}).write_parquet(path / "read_summary.parquet")


def test_disagreeing_reads_on_hand_built_reads(tmp_path):
    _sample_dir(tmp_path / "s")
    dr.main(["--sample-dir", str(tmp_path / "s"), "--out", str(tmp_path / "out"), "--n", "2"])
    summ = json.loads((tmp_path / "out" / "discordant_summary.json").read_text())
    # our_adapters_inside_sreads is 3 for 2 adapters: zm 3, a disagreeing read without an adapter inside, counts as one
    # (its empty list of inside adapters splits into one empty item), as v10 counted
    assert summ == {"pool_skera_passing_reads": 4, "disagreeing_reads": 3, "one_shifted_cut": 1, "our_adapters_inside_sreads": 3,
                    "inside_exact": 1, "inside_adapter_D": 2, "drawn_for_review": 2}
    pool = pl.read_csv(tmp_path / "out" / "discordant_all_in_pool.tsv", separator="\t")
    assert pool.columns == ["zm", "skera_gaps_without_our_adapter", "our_adapters_inside_an_sread", "inside_adapters", "gap_positions"]
    # ordered by each read's first S-read start, then by S-read table order: zm 3 (5), zm 1 (17), zm 4 (17)
    assert pool.rows() == [(3, 1, 0, "", "900-920"), (1, 1, 1, "D@1017(edits 0)", "1500-1514"), (4, 0, 1, "D'@1017(edits 1)", "")]
    picked = pl.read_csv(tmp_path / "out" / "discordant_reads_zmws.tsv", separator="\t")["zm"].to_list()
    assert len(picked) == 2 and picked == sorted(picked) and set(picked) <= {1, 3, 4}
    auto = pl.read_csv(tmp_path / "out" / "discordant_reads_auto.tsv", separator="\t")
    assert auto.columns == [*pool.columns, "read_len", "chain", "primary_pattern"]
    assert auto["zm"].to_list() == picked
    assert page_tracks((tmp_path / "out" / "discordant_reads.html").read_text()) == [str(z) for z in picked]


def page_tracks(html: str) -> list[str]:
    """The ZMW of every read track on the review page, in page order (each figure's y-axis labels), after checking that
    the page loads nothing from the network: plotly is inlined, no script or link points to a URL."""
    tags = re.findall(r"<(?:script|link|img|iframe)\b[^>]*>", html, flags=re.I)
    assert tags and not [t for t in tags if re.search(r"\b(?:src|href)=", t, flags=re.I)]
    return [z for tick in re.findall(r'"ticktext":\[([^\]]*)\]', html) for z in json.loads(f"[{tick}]")]


def test_writes_tables_and_review_page_and_has_no_hifi_option(tmp_path):
    _sample_dir(tmp_path / "s")
    with pytest.raises(SystemExit):
        dr.main(["--sample-dir", str(tmp_path / "s"), "--out", str(tmp_path / "o2"), "--hifi", "x.bam"])
    dr.main(["--sample-dir", str(tmp_path / "s"), "--out", str(tmp_path / "out")])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted([*TABLES, "discordant_reads.html"])
    html = (tmp_path / "out" / "discordant_reads.html").read_text()
    assert page_tracks(html) == ["1", "3", "4"]                     # every disagreeing read, fewer than --n


@pytest.mark.real
def test_hash_sample_tables_equal_v10_and_page_draws_the_30_reads(local, tmp_path):
    run_analysis("discordant_reads.py", "--sample-dir", str(local["hash_dir"] / "BD144_ND"), "--out", str(tmp_path / "out"))
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == sorted([*TABLES, "discordant_reads.html"])
    assert golden_diffs(out, GOLDEN_INPUTS / "discordant_reads", tmp_path / "cmp") == []
    v10 = local["hash_dir"].parent / "discordant_reads"           # v10's analysis outputs sit beside its hash sample
    for name in TABLES:
        assert (out / name).read_bytes() == (v10 / name).read_bytes(), name
    zmws = pl.read_csv(out / "discordant_reads_zmws.tsv", separator="\t")["zm"].cast(pl.Utf8).to_list()
    assert len(zmws) == 30 and page_tracks((out / "discordant_reads.html").read_text()) == zmws
