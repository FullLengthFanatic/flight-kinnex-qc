"""Tests for study/pipeline/s02_annotate_reads.py."""
import json
import random

import edlib
import polars as pl
import pytest

from kinnex_qc import runner
from kinnex_qc.patterns import FAIL_MAX_EDITS, HIFI_MAX_EDITS, MAS, R1, TSO_CORE, TSO_EXT, revcomp
from kinnex_qc.sampling import in_hash_sample
from pipeline_helpers import bam_records, mini_samples, run_stage

import s02_annotate_reads as annotate_mod


def test_stage_uses_the_package_annotate_batch():
    assert annotate_mod.annotate_batch is runner.annotate_batch


def test_token_pos():
    tp = runner.token_pos
    assert tp("C", "D") == 2
    assert tp("D'", "C'") == 2
    assert tp("C", "D'") is None
    assert tp("C", "E") is None
    assert tp(None, "A") is None
    assert tp("H", "Q") == 7


def test_tso_faces_lower():
    f = runner.tso_faces_lower
    assert f("head", "C", "D") is True
    assert f("tail", "C", "D") is False
    assert f("tail", "D'", "C'") is True
    assert f(None, "C", "D") is None


def test_fail_reads_use_relaxed_thresholds():
    """A TSO' with FAIL_MAX_EDITS["TSO"] edits is found in a fail_sub read and missed in a hifi read."""
    k = FAIL_MAX_EDITS["TSO"]
    assert k > HIFI_MAX_EDITS["TSO"]
    r = random.Random(2)

    def rand(n):
        return "".join(r.choice("ACGT") for _ in range(n))

    tso = list(revcomp(TSO_CORE))
    for i in range(2, 25, 5)[:k]:
        tso[i] = {"A": "C", "C": "G", "G": "T", "T": "A"}[tso[i]]
    tso = "".join(tso)
    assert edlib.align(revcomp(TSO_CORE), tso, mode="HW", task="distance")["editDistance"] == k

    def mol(t):
        return R1 + rand(43) + "T" * 20 + rand(600) + revcomp(TSO_EXT) + t

    seq = MAS["A"] + mol(tso) + MAS["B"] + mol(revcomp(TSO_CORE)) + MAS["C"]
    feats, _, segs = annotate_mod.annotate_batch([(1, "hifi", seq, 12, 0.999, True), (2, "fail_sub", seq, 1, 0.8, True)])
    assert [(z, e) for z, n, e in zip(feats["zm"], feats["name"], feats["edits"]) if n == "TSO"] == [(1, 0), (2, k), (2, 0)]
    assert list(zip(segs["zm"], segs["end_class"])) == [(1, "r1_only"), (1, "proper"), (2, "proper"), (2, "proper")]


@pytest.mark.real
def test_indexes_and_annotation(nd_out):
    hifi = pl.read_parquet(nd_out / "hifi_index.parquet")
    nonp = pl.read_parquet(nd_out / "non_passing_index.parquet")
    fail = pl.read_parquet(nd_out / "fail_index.parquet")
    assert (hifi.height, fail.height) == (1500, 1000)
    assert 0 < nonp.height < hifi.height
    summ = pl.read_parquet(nd_out / "read_summary.parquet")
    hs = summ.filter(pl.col("source") == "hifi")
    assert hs.height == hifi.height                          # every HiFi read is annotated
    assert (~hs["skera_passing"]).sum() == nonp.height
    assert summ.filter(pl.col("source") == "fail_sub").height > 50          # --fail-subsample 10 keeps 1 in 10
    # skera rejects the mirror arrays; the arrays it accepts are complete chains for us too
    assert (hs.filter(pl.col("primary_pattern") == "mirror")["skera_passing"] == False).mean() > 0.9  # noqa: E712
    assert (hs.filter(pl.col("skera_passing"))["primary_pattern"] == "complete").mean() > 0.98
    aseg = pl.read_parquet(nd_out / "annot_segments.parquet")
    with_pos = aseg.filter(pl.col("pos").is_not_null())
    assert with_pos.height > 0 and ((with_pos["pos"] >= 0) & (with_pos["pos"] <= 7)).all()
    complete = hs.filter(pl.col("primary_pattern") == "complete")["zm"]
    # a complete read has 8 segments between adapters; a stretch of 50 bp or more beyond A or Q adds one without a position
    ours = aseg.filter(pl.col("zm").is_in(complete) & (pl.col("source") == "hifi") & pl.col("pos").is_not_null())
    assert ours.height == 8 * complete.len()
    assert (ours["end_class"] == "proper").mean() > 0.99
    proper = ours.filter(pl.col("end_class") == "proper")
    assert proper["tso_toward_lower"].null_count() == 0


@pytest.mark.real
def test_fraction_samples_hifi_and_indexes_whole_fail_bam(mini, tmp_path):
    """--fraction 0.1: the HiFi ZMWs in the hash sample, under hash_0.1/. The fail BAM is indexed whole (1,000 mini
    records) so stage 06 knows this run's share of the cell's fail reads; 1 in 10 of them is annotated."""
    run_stage("s02_annotate_reads.py", "BD144_ND", tmp_path, mini / "samples_mini.tsv", "--fraction", "0.1",
              "--fail-subsample", "10")
    out = tmp_path / "hash_0.1" / "BD144_ND"
    zms = [r.get_tag("zm") for r in bam_records(mini_samples(mini)["BD144_ND"].hifi_bam)]
    sampled = [z for z in zms if in_hash_sample(z, 0.1)]
    assert 100 < len(sampled) < 200
    assert pl.read_parquet(out / "hifi_index.parquet")["zm"].to_list() == sampled
    assert set(pl.read_parquet(out / "non_passing_index.parquet")["zm"]) <= set(sampled)
    assert pl.read_parquet(out / "fail_index.parquet").height == 1000
    done = json.loads((out / "annotate.done.json").read_text())
    assert (done["fail"], done["fail_bam_records"], done["fail_sub_modulo"], done["fraction"]) == (1000, 1000, 10, 0.1)
    sub = pl.read_parquet(out / "read_summary.parquet").filter(pl.col("source") == "fail_sub").height
    assert 50 < sub < 150
    assert not (tmp_path / "BD144_ND").exists()
