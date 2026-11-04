"""Tests for study/analyses/chain_benchmark.py: the arrays it builds are called what they are at zero error rate, the
tables have v10's layout, and the run on the calibration S-reads equals the v10 golden files."""
import random

import polars as pl
import pytest

from kinnex_qc.annotate import HIFI_PATTERNS, adapter_chain, annotate_read, chain_flags, primary_pattern
from kinnex_qc.io import iter_bam
from pipeline_helpers import GOLDEN_INPUTS, golden_diffs, run_analysis

import chain_benchmark as cb
import gates

KNOWN = {"complete": "complete", "tso_tso_mirror_D": "mirror", "tso_tso_mirror_Q": "mirror", "r1_r1_mirror_A": "mirror",
         "r1_r1_mirror_C": "mirror", "truncated_no_A": "truncated", "truncated_no_Q": "truncated", "adapter_destroyed": "skipped_slot",
         "opposite_join": "unresolved"}


@pytest.fixture(scope="module")
def pool(test_bams):
    _, sreads = test_bams
    pool = gates._clean_sreads([r.query_sequence for r in iter_bam(sreads)])
    assert len(pool) >= 8
    return pool


@pytest.mark.parametrize("cls", KNOWN)
def test_built_arrays_are_called_what_they_are_without_errors(pool, cls):
    rng = random.Random(3)
    for _ in range(5):
        seq = cb.build(cls, pool, 0.0, rng)
        tokens, _ = adapter_chain(annotate_read(seq, HIFI_PATTERNS))
        assert primary_pattern(chain_flags(tokens)) == KNOWN[cls], tokens


def test_tso_tso_mirror_chain(pool):
    seq = cb.build("tso_tso_mirror_F", pool, 0.0, random.Random(4))
    tokens, _ = adapter_chain(annotate_read(seq, HIFI_PATTERNS))
    assert tokens == ["Q'", "H'", "G'", "F'", "F", "G", "H", "Q"]


def test_tables(test_bams, tmp_path):
    _, sreads = test_bams
    cb.main(["--sreads", str(sreads), "--out", str(tmp_path), "--n", "2"])
    t = pl.read_csv(tmp_path / "mirror_benchmark.tsv", separator="\t")
    assert t.columns[:4] == ["error_rate", "truth", "n", "called_complete"]
    assert t.height == len(cb.RATES) * 22 and set(t["n"]) == {2}
    called = [c for c in t.columns if c.startswith("called_")]
    assert (t.select(pl.sum_horizontal(called)) == 2).to_series().all()
    s = pl.read_csv(tmp_path / "mirror_benchmark_summary.tsv", separator="\t")
    assert s.columns == ["error_rate", "mirror_recall", "mirror_precision", "non_mirror_called_mirror", "non_mirror_reads"]
    assert s["error_rate"].to_list() == list(cb.RATES)
    assert set(s["non_mirror_reads"]) == {12}                        # 6 non-mirror classes x 2


@pytest.mark.real
@pytest.mark.slow
def test_calibration_sreads_equal_golden(local, tmp_path):
    run_analysis("chain_benchmark.py", "--sreads", str(local["calibration_dir"] / "nd_first200k.skera120.bam"), "--out", str(tmp_path / "out"))
    golden = GOLDEN_INPUTS / "mirror_benchmark"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(p.name for p in golden.iterdir())
    assert golden_diffs(tmp_path / "out", golden, tmp_path / "cmp") == []
