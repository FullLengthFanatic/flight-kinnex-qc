"""Tests for report.html (kinnex_qc.reader_report), the read viewer and the schematics."""
import json
import re
import subprocess

import polars as pl
import pysam
import pytest

from kinnex_qc import reader_report, runner, schematics, viewer
from kinnex_qc.patterns import KINNEX8_SC3

_IGNORE_FORK_WARNING = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")

# The fixture's six ZMWs by hash_key, smallest first: 4, 6, 2, 1, 5, 3. Every read is clean (no edits,
# rq 0.999, two primers per molecule, nothing fused). With --sreads, zm 1's S-reads do not end at our
# adapters (its false cut), so the complete examples are 2 and 3 and zm 1 is the false-cut example.
WITH_SREADS = [("complete", 2), ("complete", 3), ("tso_tso", 4), ("tso_tso", 5), ("r1_r1", 6), ("false_cut", 1)]
WITHOUT_SREADS = [("complete", 2), ("complete", 1), ("tso_tso", 4), ("tso_tso", 5), ("r1_r1", 6)]


def _run(hifi, out, *extra):
    subprocess.run(["kinnex-qc", "run", "--hifi", str(hifi), "--out", str(out), "--threads", "2", *extra],
                   capture_output=True, text=True, check=True)
    return out


def _html(out):
    return (out / "report.html").read_text(encoding="utf-8")


def _examples(html):
    return [(c, int(z)) for c, z in re.findall(r'id="example-([a-z_0-9]+)-(\d+)"', html)]


@pytest.fixture(scope="module")
def with_sreads(test_bams, tmp_path_factory):
    hifi, sreads = test_bams
    return _run(hifi, tmp_path_factory.mktemp("report_sreads"), "--sreads", str(sreads), "--fraction", "1")


@pytest.fixture(scope="module")
def without_sreads(test_bams, tmp_path_factory):
    hifi, _ = test_bams
    return _run(hifi, tmp_path_factory.mktemp("report_hifi_only"), "--fraction", "1")


def test_report_exists_with_and_without_sreads(with_sreads, without_sreads):
    for out in (with_sreads, without_sreads):
        assert (out / "report.html").is_file()
        assert _html(out).lstrip().lower().startswith("<!doctype html>")


def test_report_has_no_en_or_em_dash(with_sreads, without_sreads):
    for out in (with_sreads, without_sreads):
        html = _html(out)
        assert "\u2014" not in html
        assert "\u2013" not in html


def test_report_loads_nothing_from_the_network(with_sreads, without_sreads):
    for out in (with_sreads, without_sreads):
        html = _html(out)
        assert not re.search(r"<script\b[^>]*\bsrc\s*=", html, re.I)
        assert not re.search(r"<link\b[^>]*\bhref\s*=\s*[\"']?https?://", html, re.I)
        assert "fonts.googleapis" not in html


def test_false_cut_rate_formatted_as_in_summary_json(with_sreads):
    summary = json.loads((with_sreads / "summary.json").read_text())
    share = summary["skera"]["false_cut_reads_share"]
    html = _html(with_sreads)
    assert f"skera.false_cut_reads_share = {json.dumps(share)}" in html
    assert f"{100 * share:.1f}%" in html


def test_one_pre_per_example_with_sreads(with_sreads):
    html = _html(with_sreads)
    assert _examples(html) == WITH_SREADS
    assert html.count("<pre") == len(WITH_SREADS)


def test_without_sreads_no_skera_section_and_no_false_cut_example(without_sreads):
    html = _html(without_sreads)
    assert 'id="skera"' not in html
    assert _examples(html) == WITHOUT_SREADS
    assert html.count("<pre") == len(WITHOUT_SREADS)


def test_examples_option_sets_reads_per_category(test_bams, tmp_path):
    hifi, sreads = test_bams
    one = _html(_run(hifi, tmp_path / "one", "--sreads", str(sreads), "--fraction", "1", "--examples", "1"))
    assert _examples(one) == [("complete", 2), ("tso_tso", 4), ("r1_r1", 6), ("false_cut", 1)]
    assert one.count("<pre") == 4
    none = _html(_run(hifi, tmp_path / "none", "--sreads", str(sreads), "--fraction", "1", "--examples", "0"))
    assert _examples(none) == []
    assert "<pre" not in none


def test_report_of_zero_sampled_reads(test_bams, tmp_path):
    hifi, sreads = test_bams
    html = _html(_run(hifi, tmp_path / "out", "--sreads", str(sreads), "--fraction", "0.000001"))
    assert "<pre" not in html
    assert "\u2014" not in html and "\u2013" not in html


def test_example_reads_are_whole_reads_with_adapters_uppercase(test_bams, with_sreads):
    """Each <pre> shows its whole HiFi read: the bases, read without the position labels and tags, equal
    the record's sequence ignoring case; adapters and the false-cut site are uppercase, the rest lowercase."""
    hifi, _ = test_bams
    with pysam.AlignmentFile(str(hifi), "rb", check_sq=False) as bam:
        seqs = {r.get_tag("zm"): r.query_sequence for r in bam}
    html = _html(with_sreads)
    pres = re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.S)
    for (category, zm), pre in zip(WITH_SREADS, pres):
        shown = re.sub(r"\s", "", re.sub(r"<[^>]+>", "", re.sub(r'<span class="pos">[^<]*</span>', "", pre)))
        assert shown.upper() == seqs[zm]
        upper = "".join(re.findall(r'<span class="(?:adapter|false)"[^>]*>([^<]*)</span>', pre))
        assert upper and upper == upper.upper()
        plain = re.sub(r'<span class="(?:adapter|false)"[^>]*>[^<]*</span>', "", re.sub(r'<span class="pos">[^<]*</span>', "", pre))
        plain = re.sub(r"[\s\d]", "", re.sub(r"<[^>]+>", "", plain))
        assert plain == plain.lower()
        assert ('class="false"' in pre) == (category == "false_cut")


@_IGNORE_FORK_WARNING
def test_examples_need_the_clean_rule(test_bams):
    """A read with rq below 0.999 or an adapter with an edit is not shown: of the three complete arrays, only
    zm 1 stays once zm 2 gets rq 0.998 and one adapter of zm 3 gets one edit."""
    hifi, _ = test_bams
    read_summary, molecules, features = runner.annotate(hifi, KINNEX8_SC3, 1.0, 2)
    read_summary = read_summary.with_columns(
        pl.when(pl.col("zm") == 2).then(pl.lit(0.998, dtype=pl.Float32)).otherwise(pl.col("rq")).alias("rq"))
    first_adapter = features.filter((pl.col("zm") == 3) & (pl.col("kind") == "MAS"))["start"].min()
    features = features.with_columns(
        pl.when((pl.col("zm") == 3) & (pl.col("kind") == "MAS") & (pl.col("start") == first_adapter))
        .then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.col("edits")).alias("edits"))
    picks = reader_report.pick_examples(read_summary, molecules, features, 5, order=KINNEX8_SC3.order)
    assert [(p["category"], p["zm"]) for p in picks] == [("complete", 1), ("tso_tso", 4), ("tso_tso", 5), ("r1_r1", 6)]


def test_html_sequence_draws_adapters_on_top_with_1_based_titles():
    seq = "acgtACGTacgt"
    out = viewer.html_sequence(seq, [("A", 4, 8, "adapter"), ("TSO", 2, 6, "tso")], wrap=5)
    lines = out.split("\n")
    assert len(lines) == 3
    assert [re.match(r'<span class="pos">\s*(\d+)</span>', x).group(1) for x in lines] == ["1", "6", "11"]
    assert '<span class="tso" title="TSO (TSO), 3-6">gt</span>' in lines[0]
    assert '<span class="adapter" title="A (adapter), 5-8">A</span>' in lines[0]
    assert '<span class="adapter" title="A (adapter), 5-8">CGT</span>' in lines[1]
    assert re.sub(r"\s", "", re.sub(r"<[^>]+>", "", re.sub(r'<span class="pos">[^<]*</span>', "", out))) == seq


def test_false_cut_svg_lima_labels_only_with_fates():
    with_fates = schematics.false_cut_svg({"fates": {"fused_lima": 0.0033564735262446447, "frag_lima": 0.0}})
    assert with_fates.startswith("<svg")
    assert "lima passes 0.3% of chimeras" in with_fates
    assert "lima rejects both" in with_fates
    reader = schematics.false_cut_svg({"false_cuts": 1})
    assert reader.startswith("<svg") and reader.endswith("</svg>")
    assert "lima" not in reader
    assert schematics.schematic_svg().startswith("<svg")
