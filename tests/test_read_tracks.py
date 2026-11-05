"""Tests for study/report/read_tracks.py: the read tracks of Supplementary Fig. 3 and of the discordant-read review page,
and the choice of reads to draw."""
import polars as pl
import pytest

import pipeline_helpers  # noqa: F401  (puts study/report on sys.path)
import read_tracks as rt
from kinnex_qc.scan import Hit
from read_tracks import (ADAPTER_W, BEAD_BLOCK, BRACKET_DY, FEATURE_COLORS, LABEL_FONT_SIZE, LETTER_DY, LETTER_FONT_SIZE,
                         LETTER_STAGGER, POLYT_W, PRIMER_W, ROW_PX, TICK_DY, TICK_MARKER_SIZE, _label_bp, _subtract,
                         read_track_figure)


def read():
    hits = [Hit("A", "MAS", "+", 0, 17, 0), Hit("R1", "R1", "+", 17, 37, 0), Hit("TSO", "TSO", "-", 1500, 1525, 1),
            Hit("B", "MAS", "-", 1525, 1542, 0)]
    segs = [{"start": 17, "end": 1525, "end_class": "proper", "polyT_len": 25, "tso_side": "tail"}]
    return {"zm": 7, "read_len": 1542, "pattern": "truncated", "features": hits, "segments": segs, "skera": [(17, 1525)]}


def read_with_gap():
    """A read with an uncovered leading and trailing stretch, and a non-proper segment."""
    hits = [Hit("A", "MAS", "+", 30, 47, 0)]
    segs = [{"start": 47, "end": 150, "end_class": "tso_tso", "polyT_len": None, "tso_side": None}]
    return {"zm": 9, "read_len": 200, "pattern": "truncated", "features": hits, "segments": segs, "skera": []}


def read_close_adapters():
    """Two adapters close enough, relative to the read, that the second letter must stagger."""
    hits = [Hit("C", "MAS", "+", 4000, 4017, 0), Hit("D", "MAS", "+", 4020, 4037, 0)]
    return {"zm": 11, "read_len": 10000, "pattern": "truncated", "features": hits, "segments": [], "skera": []}


def by_name(fig, name):
    return [t for t in fig.data if t.name == name]


def test_figure_title_is_the_title_argument_not_a_legend_group_title():
    """The loop that adds the legend entries must not overwrite the title argument."""
    assert read_track_figure([read()], "Reads 1 to 1 of 1").layout.title.text == "Reads 1 to 1 of 1"
    assert read_track_figure([read()], "").layout.title.text == ""


def test_track_features_and_widths():
    fig = read_track_figure([read()], "t")
    r1 = [t for t in fig.data if t.name == "R1" and t.type == "bar"][0]
    assert list(r1.base) == [17] and list(r1.x) == [20 + BEAD_BLOCK]
    polyt = [t for t in fig.data if t.name == "polyT" and t.type == "bar"][0]
    assert list(polyt.base) == [37 + BEAD_BLOCK] and list(polyt.x) == [25]
    a = [t for t in fig.data if t.name == "A" and t.type == "bar"][0]
    assert a.x[0] == max(17, 1542 / 300)                     # narrow adapters widened to stay visible
    b = [t for t in fig.data if t.name == "B" and t.type == "bar"][0]
    assert list(b.marker.pattern.shape) == ["/"] and b.marker.color == FEATURE_COLORS["B"]
    assert by_name(fig, "hatched: adapter on the minus strand")
    sk = by_name(fig, "skera S-read boundary")[0]
    assert list(sk.x) == [17, 1525]
    assert list(sk.y) == [0 + TICK_DY, 0 + TICK_DY]            # ticks sit below the bar, not above


def test_track_no_class_band():
    fig = read_track_figure([read()], "t")
    assert not [t for t in fig.data if t.name and t.name.startswith("segment:")]


def test_track_cdna_and_feature_hover_carry_segment_class():
    fig = read_track_figure([read()], "t")
    cdna = [t for t in fig.data if t.name == "cDNA" and t.type == "bar"][0]
    assert "segment: proper" in cdna.hovertext[0]
    tso = [t for t in fig.data if t.name == "TSO" and t.type == "bar"][0]
    assert "segment: proper" in tso.hovertext[0]
    r1 = [t for t in fig.data if t.name == "R1" and t.type == "bar"][0]
    assert "segment: proper" in r1.hovertext[0]


def test_track_unannotated_for_uncovered_stretch():
    fig = read_track_figure([read(), read_with_gap()], "t")
    un = [t for t in fig.data if t.name == "unannotated" and t.type == "bar"][0]
    assert list(un.base) == [0, 150] and list(un.x) == [30, 50]
    assert un.marker.line.width and un.marker.line.color == "#52514e"
    assert by_name(fig, "unannotated")  # legend entry present


def test_track_no_unannotated_when_fully_covered():
    fig = read_track_figure([read()], "t")
    assert not [t for t in fig.data if t.name == "unannotated"]


def test_track_labels_non_proper_segments():
    fig = read_track_figure([read(), read_with_gap()], "t")
    labels = [t for t in fig.data if getattr(t, "mode", None) == "text" and t.textfont.color == "#52514e"][0]
    assert "<b>tso_tso</b>" in list(labels.text) and "<b>proper</b>" not in list(labels.text)
    row = 1                                                   # read_with_gap() is the second read
    assert list(labels.y) == [row + BRACKET_DY]               # label centred on the bracket line
    bracket = [t for t in fig.data if getattr(t, "mode", None) == "lines"][0]
    start, end = 47, 150                                      # read_with_gap()'s non-proper segment
    mid = (start + end) / 2
    half = _label_bp("tso_tso", 1542) / 2                     # 1542 bp: the longest read in the figure
    assert list(bracket.x) == pytest.approx([start, mid - half, None, mid + half, end, None])  # cut around the label
    y = row + BRACKET_DY
    assert list(bracket.y) == [y, y, None, y, y, None]
    assert ADAPTER_W / 2 < TICK_DY < BRACKET_DY                # tick clears the bar, bracket clears the tick
    assert ADAPTER_W / 2 < LETTER_DY                           # letter (above) clears the bar the same way


def test_subtract():
    assert _subtract((0, 100), [(40, 60)]) == [(0, 40), (60, 100)]
    assert _subtract((0, 100), [(-10, 120)]) == []
    assert _subtract((0, 100), [(150, 160), (-20, -10)]) == [(0, 100)]


def short_labels_read():
    """Two short non-proper segments side by side: their labels would overlap, so they merge into one,
    and no bracket piece runs under the merged label."""
    hits = [Hit("B", "MAS", "-", 1000, 1017, 0), Hit("B", "MAS", "+", 1117, 1134, 0), Hit("C", "MAS", "+", 1234, 1251, 0)]
    segs = [{"start": 17, "end": 1000, "end_class": "proper", "polyT_len": None, "tso_side": None},
            {"start": 1017, "end": 1117, "end_class": "tso_tso", "polyT_len": None, "tso_side": None},
            {"start": 1134, "end": 1234, "end_class": "r1_only", "polyT_len": None, "tso_side": None},
            {"start": 1251, "end": 10000, "end_class": "proper", "polyT_len": None, "tso_side": None}]
    return {"zm": 13, "read_len": 10000, "pattern": "mirror", "features": hits, "segments": segs, "skera": []}


def test_touching_labels_merge_and_brackets_avoid_text():
    fig = read_track_figure([short_labels_read()], "t")
    labels = [t for t in fig.data if getattr(t, "mode", None) == "text" and t.textfont.color == "#52514e"][0]
    assert list(labels.text) == ["<b>tso_tso, r1_only</b>"]
    x = labels.x[0]
    half = _label_bp("tso_tso, r1_only", 10000) / 2
    lines = [t for t in fig.data if getattr(t, "mode", None) == "lines"]
    xs = [v for v in lines[0].x if v is not None] if lines else []
    for a, b in zip(xs[::2], xs[1::2]):
        assert b <= x - half or a >= x + half                  # every bracket piece stays clear of the label


def test_three_crowded_letters_use_two_levels_then_join():
    hits = [Hit("H", "MAS", "+", 5000, 5017, 0), Hit("Q", "MAS", "-", 5030, 5047, 0), Hit("Q", "MAS", "+", 5060, 5077, 0)]
    r = {"zm": 15, "read_len": 10000, "pattern": "mirror", "features": hits, "segments": [], "skera": []}
    fig = read_track_figure([r], "t")
    letters = [t for t in fig.data if getattr(t, "mode", None) == "text" and t.textfont.color == "#0b0b0b"][0]
    assert list(letters.text) == ["H", "Q' Q"]                 # the third letter joins the lifted one
    assert list(letters.y) == pytest.approx([-LETTER_DY, -LETTER_DY - LETTER_STAGGER])


def test_row_layout_has_min_gap():
    """The per-row vertical stack (a row's staggered letter, its letter, its bar, its tick, its bracket+label, then the
    next row's staggered letter) must never overlap and must clear by at least 0.03 row units, computed from the module
    constants, so a later constant change cannot make a staggered letter land under the class label of the row above."""
    MIN_GAP = 0.03
    letter_h = 1.2 * LETTER_FONT_SIZE / ROW_PX      # text height in px is about 1.2x font size
    label_h = 1.2 * LABEL_FONT_SIZE / ROW_PX
    tick_h = TICK_MARKER_SIZE / ROW_PX
    r = 0
    stack = [
        ("staggered letter", r - LETTER_DY - LETTER_STAGGER, letter_h),
        ("letter", r - LETTER_DY, letter_h),
        ("bar", r, ADAPTER_W),
        ("tick", r + TICK_DY, tick_h),
        ("bracket+label", r + BRACKET_DY, label_h),
        ("next row's staggered letter", (r + 1) - LETTER_DY - LETTER_STAGGER, letter_h),
    ]
    intervals = [(name, c - h / 2, c + h / 2) for name, c, h in stack]
    for (name_a, _, bottom_a), (name_b, top_b, _) in zip(intervals, intervals[1:]):
        gap = top_b - bottom_a
        assert gap >= MIN_GAP, f"{name_a} to {name_b}: gap {gap:.4f} < {MIN_GAP}"


def test_track_heights_adapter_vs_primer_vs_polyt():
    fig = read_track_figure([read()], "t")
    a = [t for t in fig.data if t.name == "A" and t.type == "bar"][0]
    tso = [t for t in fig.data if t.name == "TSO" and t.type == "bar"][0]
    r1 = [t for t in fig.data if t.name == "R1" and t.type == "bar"][0]
    polyt = [t for t in fig.data if t.name == "polyT" and t.type == "bar"][0]
    cdna = [t for t in fig.data if t.name == "cDNA" and t.type == "bar"][0]
    assert a.width == ADAPTER_W == cdna.width
    assert tso.width == r1.width == PRIMER_W
    assert polyt.width == POLYT_W
    assert POLYT_W < PRIMER_W < ADAPTER_W                      # adapter > primer > polyT


def test_track_adapter_letters_with_prime_for_minus_strand():
    fig = read_track_figure([read()], "t")
    letters = [t for t in fig.data if getattr(t, "mode", None) == "text" and t.textfont.color == "#0b0b0b"][0]
    assert list(letters.text) == ["A", "B'"]                   # A is '+', B is '-'
    assert list(letters.y) == [0 - LETTER_DY, 0 - LETTER_DY]   # both features are on the single read (row 0)
    assert LETTER_DY > ADAPTER_W / 2                           # clear of the bar top, no overlap
    assert letters.textfont.size == LETTER_FONT_SIZE and 12 <= LETTER_FONT_SIZE <= 13  # readable at 50 reads


def test_track_adapter_letters_stagger_when_close():
    fig = read_track_figure([read_close_adapters()], "t")
    letters = [t for t in fig.data if getattr(t, "mode", None) == "text" and t.textfont.color == "#0b0b0b"][0]
    by_x = sorted(zip(letters.x, letters.y, letters.text))
    (x0, y0, t0), (x1, y1, t1) = by_x
    assert (t0, t1) == ("C", "D")
    assert y0 == 0 - LETTER_DY                                  # first letter at the base height
    assert y1 == 0 - LETTER_DY - LETTER_STAGGER                 # close neighbour lifted an extra notch


def test_q_colour_distinct_from_every_other_feature_colour():
    assert len(set(FEATURE_COLORS.values())) == len(FEATURE_COLORS)
    assert FEATURE_COLORS["Q"] != FEATURE_COLORS["TSO"]


def test_legend_groups_and_order():
    fig = read_track_figure([read()], "t")
    # Vertical, pinned to the right: a horizontal legend can wrap a group's title and its first entry onto
    # their own line, breaking rank order. A single column has no wrap point.
    assert fig.layout.legend.orientation == "v"
    assert fig.layout.legend.x >= 1
    markers = {t.name: t for t in fig.data if getattr(t, "mode", None) == "markers" and t.marker.symbol == "square"}
    assert markers["A"].legendgrouptitle.text == "Adapters"
    assert markers["TSO primer"].legendgrouptitle.text == "Primers"
    assert markers["Read1 + cell label/UMI"].legendgrouptitle.text in (None, "")
    ranks = {t.name: t.legendrank for t in fig.data if getattr(t, "legendrank", None) is not None}
    assert ranks["A"] < ranks["B"] < ranks["TSO primer"] < ranks["Read1 + cell label/UMI"] \
        < ranks["polyT"] < ranks["cDNA"] < ranks["skera S-read boundary"] \
        < ranks["hatched: adapter on the minus strand"]
    assert markers["cDNA"].legendgrouptitle.text == "Insert and marks"


def test_legend_adapters_title_without_a():
    """A TSO-TSO read rarely has an A adapter; the 'Adapters' title goes on whichever adapter is drawn."""
    hits = [Hit("B", "MAS", "+", 0, 17, 0), Hit("C", "MAS", "+", 1500, 1517, 0), Hit("TSO", "TSO", "-", 700, 725, 1)]
    r = {"zm": 5, "read_len": 1600, "pattern": "mirror", "features": hits, "segments": [], "skera": []}
    fig = read_track_figure([r], "t")
    markers = {t.name: t for t in fig.data if getattr(t, "mode", None) == "markers" and t.marker.symbol == "square"}
    assert "A" not in markers
    assert markers["B"].legendgrouptitle.text == "Adapters"
    assert markers["C"].legendgrouptitle.text in (None, "")


def test_track_empty():
    assert len(read_track_figure([], "t").data) == 0


def frames():
    summ = pl.DataFrame({"zm": [1, 2, 3, 4], "source": ["hifi"] * 4, "read_len": [9000] * 4, "chain": ["A B"] * 4,
                         "primary_pattern": ["complete", "complete", "mirror", "complete"],
                         "skera_passing": [True, True, False, True]}, schema_overrides={"zm": pl.UInt32})
    seg_rows = []
    for zm in (1, 2, 4):
        for i in range(8):
            seg_rows.append({"zm": zm, "source": "hifi", "start": 100 * i, "end": 100 * i + 90, "pos": i,
                             "end_class": "tso_tso" if (zm == 2 and i == 3) else "proper", "internal": False, "polyT_len": 20, "tso_side": "tail"})
    seg_rows.append({"zm": 3, "source": "hifi", "start": 0, "end": 90, "pos": None, "end_class": "tso_tso", "internal": False,
                     "polyT_len": None, "tso_side": None})
    segs = pl.DataFrame(seg_rows, schema_overrides={"zm": pl.UInt32})
    mas = pl.DataFrame({"zm": [4, 1], "source": ["hifi", "hifi"], "kind": ["MAS", "MAS"], "start": [400, 90], "end": [417, 100]},
                       schema_overrides={"zm": pl.UInt32})
    skera = pl.DataFrame({"zm": [4, 1], "qs": [300, 0], "qe": [700, 90]}, schema_overrides={"zm": pl.UInt32})
    return summ, segs, mas, skera


def test_strata():
    g = rt.strata(*frames())
    assert g["complete_proper"] == [1, 4]
    assert g["tso_tso"] == [2, 3]
    assert g["skera_rejected_mirror"] == [3]
    assert g["skera_missplit"] == [4]


def test_sample_is_deterministic():
    zms = list(range(1000))
    assert rt.sample(zms, 50) == rt.sample(list(reversed(zms)), 50)
    assert len(rt.sample(zms, 50)) == 50 and rt.sample([3, 1], 50) == [1, 3]


def test_tracks_in_zmw_order_with_hits_segments_and_sread_bounds():
    summ, segs, _, skera = frames()
    feats = pl.DataFrame({"zm": [4, 4, 1], "source": "hifi", "name": ["B", "TSO", "A"], "kind": ["MAS", "TSO", "MAS"],
                          "strand": ["+", "-", "+"], "start": [400, 300, 90], "end": [417, 325, 100], "edits": [0, 1, 0]},
                         schema_overrides={"zm": pl.UInt32})
    t = rt.tracks([4, 1, 3], summ, segs, feats, skera)
    assert [r["zm"] for r in t] == [1, 3, 4]
    assert t[0]["features"] == [Hit("A", "MAS", "+", 90, 100, 0)] and t[1]["features"] == []
    assert t[2]["features"] == [Hit("B", "MAS", "+", 400, 417, 0), Hit("TSO", "TSO", "-", 300, 325, 1)]
    assert t[0]["skera"] == [(0, 90)] and t[1]["skera"] == [] and t[2]["skera"] == [(300, 700)]
    assert [s["start"] for s in t[2]["segments"]] == [100 * i for i in range(8)]
    assert t[1]["segments"] == [{"start": 0, "end": 90, "end_class": "tso_tso", "polyT_len": None, "tso_side": None}]
