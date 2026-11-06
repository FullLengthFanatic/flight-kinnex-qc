"""Tests for report.html (kinnex_qc.reader_report), the read viewer and the schematics."""
import heapq
import json
import re
import subprocess
import tracemalloc
from pathlib import Path

import numpy as np
import plotly
import polars as pl
import pysam
import pytest

from kinnex_qc import reader_report, runner, schematics, skera, viewer
from kinnex_qc.patterns import KINNEX8_SC3
from kinnex_qc.sampling import hash_key

_IGNORE_FORK_WARNING = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
GOLDEN = Path(__file__).parent / "golden" / "v10_inputs"

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


@pytest.fixture(scope="module")
def annotated(test_bams):
    hifi, _ = test_bams
    return runner.annotate(hifi, KINNEX8_SC3, 1.0, 2)


@_IGNORE_FORK_WARNING
def test_examples_clean_rule_fused_molecule_primer_edits_and_missing_primer(annotated):
    """zm 2, the first complete array by hash, is not shown once one of its molecules is fused, one of its primers
    has 2 edits, or one of its primers is missing; zm 1 and 3 still are. A primer with 1 edit is allowed."""
    read_summary, molecules, features = annotated
    first_primer = features.filter((pl.col("zm") == 2) & (pl.col("kind") != "MAS"))["start"].min()
    zm2_primer = (pl.col("zm") == 2) & (pl.col("kind") != "MAS") & (pl.col("start") == first_primer)
    first_molecule = (pl.col("zm") == 2) & (pl.col("start") == molecules.filter(pl.col("zm") == 2)["start"].min())

    def primer_edits(n):
        return features.with_columns(pl.when(zm2_primer).then(pl.lit(n, dtype=pl.Int8)).otherwise(pl.col("edits")).alias("edits"))

    cases = {
        "fused molecule": (molecules.with_columns(pl.when(first_molecule).then(True).otherwise(pl.col("internal")).alias("internal")),
                           features),
        "primer with 2 edits": (molecules, primer_edits(2)),
        "missing primer": (molecules, features.filter(~zm2_primer)),
    }
    for case, (mols, feats) in cases.items():
        picks = reader_report.pick_examples(read_summary, mols, feats, 5, order=KINNEX8_SC3.order)
        assert [(p["category"], p["zm"]) for p in picks] == [("complete", 1), ("complete", 3), *WITHOUT_SREADS[2:]], case
    picks = reader_report.pick_examples(read_summary, molecules, primer_edits(1), 5, order=KINNEX8_SC3.order)
    assert [(p["category"], p["zm"]) for p in picks][:3] == [("complete", 2), ("complete", 1), ("complete", 3)]


@_IGNORE_FORK_WARNING
@pytest.mark.parametrize("overlap", [0, 10])
def test_false_cut_examples_leave_out_touching_or_overlapping_sreads(test_bams, annotated, overlap):
    """A false cut whose two S-reads touch (overlap 0) or overlap has no gap to mark, so its read is not picked:
    zm 1, the fixture's only false cut, drops out once its D-E S-read starts where, or before, the C-D one ends."""
    _, sreads_bam = test_bams
    read_summary, molecules, features = annotated
    sreads = runner.read_sreads(sreads_bam, 1)
    qe_before = sreads.filter((pl.col("zm") == 1) & (pl.col("di") == 2))["qe"][0]
    sreads = sreads.with_columns(pl.when((pl.col("zm") == 1) & (pl.col("di") == 3))
                                 .then(pl.lit(qe_before - overlap, dtype=pl.Int32)).otherwise(pl.col("qs")).alias("qs"))
    fc = skera.false_cuts(sreads, features)
    cut = fc.filter(pl.col("zm") == 1).row(0, named=True)
    assert fc.filter(pl.col("zm") == 1).height == 1 and cut["cut_end"] - cut["cut_start"] == -overlap
    passing = skera.passing_from_labels(sreads, order_len=len(KINNEX8_SC3.order))
    picks = reader_report.pick_examples(read_summary, molecules, features, 5, order=KINNEX8_SC3.order,
                                        sreads=sreads, passing=passing, fc=fc)
    assert [p["zm"] for p in picks if p["category"] == "false_cut"] == []


def test_first_by_hash_orders_as_python_sorting_by_hash_key():
    """_first_by_hash orders its candidates as a numpy array (hash_key_array); its picks must be those of sorting the
    distinct ZMWs as Python ints by hash_key, with and without an ok filter (checked in blocks of _CHUNK, given as
    lists of Python ints), and none for no candidates."""
    rng = np.random.default_rng(5)
    zms = rng.integers(0, 2**32, 20_000, dtype=np.uint64).astype(np.uint32)
    series = pl.Series("zm", np.concatenate([zms, zms[:3000]]), dtype=pl.UInt32)
    by_hash = sorted(set(zms.tolist()), key=hash_key)
    blocks = []

    def ok(block):
        blocks.append(block)
        return {z for z in block if z % 97 == 0}

    for n in (1, 2, 5, 25_000):
        assert reader_report._first_by_hash(series, n) == by_hash[:n]
        blocks.clear()
        assert reader_report._first_by_hash(series, n, ok=ok) == [z for z in by_hash if z % 97 == 0][:n]
        assert blocks[0] == by_hash[:reader_report._CHUNK] and all(type(z) is int for b in blocks for z in b)
    assert reader_report._first_by_hash(series.clear(), 2) == []
    assert reader_report._first_by_hash(series.clear(), 2, ok=ok) == []


def _clean_complete_reads(n):
    """n reads that pass the clean rule, each a complete two-adapter array (A, Read1, TSO, B) holding one proper
    molecule: the columns pick_examples reads, built in polars, so the test holds no Python list of the ZMWs."""
    zm = pl.DataFrame(pl.int_range(1, n + 1, dtype=pl.UInt32, eager=True).alias("zm"))
    read_summary = zm.with_columns(pl.lit(1032, dtype=pl.Int32).alias("read_len"), pl.lit("complete").alias("primary_pattern"),
                                   pl.lit(2, dtype=pl.Int16).alias("n_mas"), pl.lit(1.0, dtype=pl.Float32).alias("rq"))
    molecules = zm.with_columns(pl.lit(16, dtype=pl.Int32).alias("start"), pl.lit(1000, dtype=pl.Int32).alias("end"),
                                pl.lit("A").alias("left_token"), pl.lit("B").alias("right_token"),
                                pl.lit("proper").alias("end_class"), pl.lit(False).alias("internal"))
    features = pl.concat([zm.with_columns(pl.lit(name).alias("name"), pl.lit(kind).alias("kind"), pl.lit("+").alias("strand"),
                                          pl.lit(start, dtype=pl.Int32).alias("start"), pl.lit(start + 16, dtype=pl.Int32).alias("end"),
                                          pl.lit(0, dtype=pl.Int8).alias("edits"))
                          for name, kind, start in (("A", "MAS", 0), ("R1", "R1", 16), ("TSO", "TSO", 984), ("B", "MAS", 1000))])
    return read_summary, molecules, features


def test_pick_examples_keeps_candidate_zmws_out_of_python_lists():
    """pick_examples orders its candidate ZMWs as numpy arrays, not as Python ints. 300,000 clean complete reads: as
    Python lists, sets and sort keys the candidates took a Python heap peak of 38 MB (measured before the change;
    7.2 MB after); the two picks are the first two ZMWs by hash_key either way."""
    n = 300_000
    read_summary, molecules, features = _clean_complete_reads(n)
    tracemalloc.start()
    try:
        picks = reader_report.pick_examples(read_summary, molecules, features, 2, order=KINNEX8_SC3.order)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert [(p["category"], p["zm"]) for p in picks] == [("complete", z) for z in heapq.nsmallest(2, range(1, n + 1), key=hash_key)]
    assert peak < 15 * 2**20


def _pct_span(values) -> str:
    """Whole percentages over the three libraries as the report text gives them: "74%" when they round alike, else "74 to 75%"."""
    lo, hi = round(100 * min(values)), round(100 * max(values))
    return f"{lo}%" if lo == hi else f"{lo} to {hi}%"


def test_bd144_numbers_in_the_text_match_the_golden_files(with_sreads):
    """The BD144 observations the report states as fixed text, recomputed from the golden files they come from."""
    fa = [json.loads((GOLDEN / "false_cut_anatomy" / f"BD144_{lib}_false_cuts.json").read_text()) for lib in ("ND", "HM", "kin")]
    arms = pl.read_csv(GOLDEN / "skera_mirror_arms" / "nd_hash_skera26_molecules.tsv", separator="\t")
    clean = {r: arms.filter((pl.col("region") == r) & (pl.col("class") == "CLEAN"))["share"][0] for r in ("after", "before")}
    html = re.sub(r"\s+", " ", _html(with_sreads))
    assert f"skera labelled the cut D in {_pct_span([f['skera_calls_D'] for f in fa])} of false cuts in each library" in html
    found = {f"{100 * f['missed_adapter_found']:.1f}" for f in fa}
    assert len(found) == 1 and f"found at {found.pop()}% of false cuts" in html
    assert f"was D in {_pct_span([f['missed_is_D'] for f in fa])} of those" in html
    assert f"the letter skera gave the cut in {_pct_span([f['missed_same_letter_as_skera_call'] for f in fa])}" in html
    assert {round(f["median_cut_to_missed_adapter_bp"], -2) for f in fa} == {900} and "a median of about 900 bp past it" in html
    assert max(f["fates"]["fused_lima"] for f in fa) < 0.005 and "in under 0.5% of cases" in html
    assert max(f["fates"]["frag_lima"] for f in fa) < 0.0001 and "almost never" in html
    assert f"skera 26.1.0 wrote {round(100 * clean['after'])}% and {round(100 * clean['before'])}% of them cleanly" in html


def test_false_cut_caption_labels_the_drawing_as_observed(with_sreads):
    caption = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", re.sub(r"\s+", " ", _html(with_sreads))).group(1)
    assert "are as observed in the three BD144 libraries (skera 26.1.0)" in caption


@pytest.mark.parametrize("fmt, value, shown", [
    (reader_report._pct, 0.0, "0%"), (reader_report._pct, 1e-07, "< 0.01%"), (reader_report._pct, 0.0042, "0.42%"),
    (reader_report._pct, 0.3333333333333333, "33.3%"),
    (reader_report._num, 0.0, "0"), (reader_report._num, 7.5e-05, "< 0.01"), (reader_report._num, 4.166666666666667, "4.17"),
    (reader_report._num, -0.5, "-0.50"), (reader_report._num, 1234, "1,234"),
    (lambda v: reader_report._num(v, 3), 0.0004, "< 0.001"), (lambda v: reader_report._num(v, 3), 0.0892, "0.089"),
    (reader_report._per1000, 0.0, "0"), (reader_report._per1000, 7.5e-05, "< 0.01"), (reader_report._per1000, 0.5, "0.50"),
    (reader_report._per1000, 60.60606060606061, "60.6"),
])
def test_number_formats_are_fixed_point_with_a_floor(fmt, value, shown):
    assert fmt(value) == shown


def test_slot_budget_other_molecules_have_a_row_and_a_legend(with_sreads, tmp_path):
    """'Other molecules' (molecules - usable - artifact, no summary.json key of its own) has a row in the slot
    budget's numbers table and a sentence under the bar."""
    summary = json.loads((with_sreads / "summary.json").read_text())
    b = summary["skera"]["budget"]
    assert b["molecules"] - b["usable"] - b["artifact"] == 0           # the fixture has no molecule of the "other" class
    summary["skera"]["budget"] = {**b, "molecules": b["molecules"] + 0.25}
    for page, other in ((_html(with_sreads), "0"), (reader_report.write(tmp_path, summary, []).read_text(encoding="utf-8"), "0.25")):
        assert re.search(rf'<td><span class="val">{re.escape(other)}</span></td><td>Other molecules', page)
        assert "molecules minus usable minus artifact molecules" in re.sub(r"\s+", " ", page)


def test_plotly_is_inlined_once(with_sreads, without_sreads):
    banner = f"* plotly.js v{plotly.offline.get_plotlyjs_version()}"
    for out in (with_sreads, without_sreads):
        assert _html(out).count(banner) == 1


def _software(html):
    return re.search(r"<td>Software</td><td>(.*?)</td>", html).group(1)


def test_software_row_names_skera_only_with_sreads(with_sreads, without_sreads):
    assert "skera 26.1.0 (from the S-read BAM's header)" in _software(_html(with_sreads))
    assert "skera" not in _software(_html(without_sreads))


def test_molecules_written_row_says_it_counts_every_read(with_sreads, tmp_path):
    """--write-molecules writes the molecules of every read in the file while the other numbers describe the hash
    sample, so its row says where its number comes from."""
    summary = json.loads((with_sreads / "summary.json").read_text())
    summary["molecules_written"] = 30
    page = reader_report.write(tmp_path, summary, []).read_text(encoding="utf-8")
    assert "<td>Molecules written to molecules.bam, from every read in the file</td><td>30</td>" in page


@pytest.mark.parametrize("molecules_written", [None, 30])
def test_inputs_paragraph_names_the_numbers_that_count_the_whole_file(with_sreads, tmp_path, molecules_written):
    """Below a hash sample's inputs table: the reads in the file (also under the title) and, with --write-molecules,
    the molecules written count the whole file; every other number describes the sample."""
    summary = json.loads((with_sreads / "summary.json").read_text())
    summary["fraction"] = 0.1
    whole = '"HiFi reads in the file" (in this table and under the title) counts'
    if molecules_written is not None:
        summary["molecules_written"] = molecules_written
        whole = ('"HiFi reads in the file" (in this table and under the title) and "Molecules written to molecules.bam" '
                 'count')
    page = reader_report.write(tmp_path, summary, []).read_text(encoding="utf-8")
    assert "every number above describes that sample" not in page
    assert (f"whatever the order of the file. Only {whole} the whole file; every other number above describes that "
            "sample.") in page


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


def test_report_words_the_size_rule_and_the_false_cut_as_the_note_does(with_sreads):
    """The mirror size rule in words, the chimera, the real adapter skera did not cut at and the reads that cover the
    array's reverse complement, in the words of the Supplementary Note and docs/ (the drawing is the Note's Fig. 1)."""
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", _html(with_sreads)))
    new = ["A whole two-armed mirror with its junction at adapter X therefore holds one molecule per slot from X to the last adapter on "
           "each of its 2 arms, plus 1 TSO-TSO molecule.",
           "lima passed the chimera (a whole molecule, then the real adapter skera did not cut at, then the start of the next molecule)",
           "with the Kinnex 8-fold letters and D as the real adapter skera did not cut at.",
           "the strand rule (on reads that cover the array's reverse complement, adapters Q to A, the fragment keeps the Read1 end)",
           "On reads that cover the array's reverse complement (adapters Q to A) the picture is mirrored. The cut falls in the molecule "
           "after the real adapter skera did not cut at, in the direction of the read, and the fragment keeps the Read1 end."]
    assert [s for s in new if s not in text] == []
    old = ["2 x (slots from X to the last adapter)", "reads from the other strand", "before the skipped adapter", "D as the skipped adapter"]
    assert [s for s in old if s in text] == []
