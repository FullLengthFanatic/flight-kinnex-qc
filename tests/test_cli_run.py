"""Tests for `kinnex-qc run`."""
import csv
import gzip
import json
import subprocess

import polars as pl

from kinnex_qc import runner
from kinnex_qc.annotate import FAIL_PATTERNS, HIFI_PATTERNS
from kinnex_qc.patterns import KINNEX8_SC3


def _run(*args, check=False):
    return subprocess.run(["kinnex-qc", "run", *args], capture_output=True, text=True, check=check)


def _reads_tsv(path):
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_default_chemistry_patterns_match_annotate_module():
    """For the default chemistry, runner's per-chemistry pattern cache must equal the study's module-level
    HIFI_PATTERNS and FAIL_PATTERNS (same thresholds, same adapters)."""
    assert runner._hifi_patterns(KINNEX8_SC3) == HIFI_PATTERNS
    assert runner._fail_patterns(KINNEX8_SC3) == FAIL_PATTERNS


def test_annotate_molecule_end_classes(test_bams):
    """Sanity check of the test BAM fixture itself: the mirror junctions come out as built."""
    hifi, _ = test_bams
    read_summary, molecules, _ = runner.annotate(hifi, KINNEX8_SC3, 1.0, 2)
    assert read_summary.sort("zm")["primary_pattern"].to_list() == ["complete", "complete", "complete",
                                                                    "mirror", "mirror", "mirror"]
    tso_tso = molecules.filter(pl.col("zm").is_in([4, 5]) & (pl.col("left_token") == "H'") & (pl.col("right_token") == "H"))
    assert tso_tso["end_class"].to_list() == ["tso_tso", "tso_tso"]
    r1_r1 = molecules.filter((pl.col("zm") == 6) & (pl.col("left_token") == "B") & (pl.col("right_token") == "B'"))
    assert r1_r1["end_class"].to_list() == ["r1_r1"]


def test_run_writes_reads_tsv_with_pattern_counts(test_bams, tmp_path):
    hifi, _ = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", "1", "--threads", "2", check=True)
    assert r.returncode == 0

    rows = _reads_tsv(out / "reads.tsv.gz")
    assert len(rows) == 6
    assert set(rows[0]) == {"zm", "read_len", "chain", "chain_flipped", "primary_pattern", "n_mas"}
    patterns = [row["primary_pattern"] for row in rows]
    assert patterns.count("complete") == 3
    assert patterns.count("mirror") == 3

    summary = json.loads((out / "summary.json").read_text())
    assert summary["hifi_reads_in_file"] == 6
    assert summary["hifi_reads"] == 6
    assert summary["fraction"] == 1.0
    assert summary["skera_version"] is None
    assert (out / "run.log").exists()


def test_fraction_zero_and_above_one_exit_2(test_bams, tmp_path):
    hifi, _ = test_bams
    for bad in ("0", "1.5"):
        r = _run("--hifi", str(hifi), "--out", str(tmp_path / bad), "--fraction", bad, "--threads", "2")
        assert r.returncode == 2
        assert r.stderr.startswith("kinnex-qc run: error:")


def test_write_molecules_without_fraction_sets_fraction_to_one(test_bams, tmp_path):
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--write-molecules", "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert "fraction set to 1" in r.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["fraction"] == 1.0


def test_write_molecules_without_sreads_or_template_exits_2(test_bams, tmp_path):
    hifi, _ = test_bams
    r = _run("--hifi", str(hifi), "--out", str(tmp_path / "out"), "--write-molecules", "--threads", "2")
    assert r.returncode == 2
    assert r.stderr.startswith("kinnex-qc run: error:")


def test_summary_records_skera_version_with_sreads(test_bams, tmp_path):
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "1", "--threads", "2",
              check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["skera_version"] == "26.1.0"
