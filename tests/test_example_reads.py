"""Tests for examples/make_example_reads.py: the five example reads of BD144_ND. The real test regenerates the files from
the calibration set and the hash sample and compares the FASTA byte for byte with the tracked one."""
import re

import pytest

from kinnex_qc import reader_report, viewer
from pipeline_helpers import REPO, run_script

EXAMPLES = REPO / "examples"
SCRIPT = EXAMPLES / "make_example_reads.py"
BANNED = ["leverage", "utilize", "robust", "seamless", "crucial", "delve", "harness", "streamline", "genuinely"]


def test_reader_report_exposes_the_clean_and_adapter_rules():
    assert callable(reader_report.clean) and callable(reader_report.at_adapters)


def test_html_sequence_takes_its_own_labels():
    seq = "acgtACGT"
    labels = {**viewer.LABEL, "false": "site skera called adapter D"}
    out = viewer.html_sequence(seq, [("skera_D", 2, 5, "false")], labels=labels)
    assert 'title="skera_D (site skera called adapter D), 3-5"' in out
    assert 'title="skera_D (site skera called an adapter), 3-5"' in viewer.html_sequence(seq, [("skera_D", 2, 5, "false")])


def test_tracked_example_files_are_five_reads_in_plain_text():
    fasta = (EXAMPLES / "example_reads.fasta").read_text()
    heads = [x for x in fasta.splitlines() if x.startswith(">")]
    assert [h[1:].split()[0] for h in heads] == ["complete_array", "TSO-TSO_mirror", "R1-R1_mirror", "double_mirror", "skera_false_cut"]
    assert all("BD144_ND" in h for h in heads)
    for p in (SCRIPT, EXAMPLES / "example_reads.fasta", EXAMPLES / "example_reads.html"):
        text = p.read_text()
        assert "—" not in text and "–" not in text, p.name
        assert not re.search(r"(?:^|[\s\"'=(])/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text), p.name
        assert not [w for w in BANNED if w in text.lower()], p.name


@pytest.mark.real
def test_regenerated_fasta_equals_the_tracked_one(local, tmp_path):
    run_script(SCRIPT, "--calibration-dir", str(local["calibration_dir"]), "--hash-sample-dir", str(local["hash_dir"] / "BD144_ND"),
               "--out", str(tmp_path))
    assert (tmp_path / "example_reads.fasta").read_bytes() == (EXAMPLES / "example_reads.fasta").read_bytes()
    assert (tmp_path / "example_reads.html").read_bytes() == (EXAMPLES / "example_reads.html").read_bytes()
