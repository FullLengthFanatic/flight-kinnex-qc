"""Read tracks: one horizontal track per HiFi read, to scale, with our adapters, primers and molecules and skera's S-read
boundaries. Used by Supplementary Fig. 3 (figures.fig3) and by the review page of study/analyses/discordant_reads.py.
strata and sample pick reads to draw; tracks gathers the track data of given reads from a sample's pipeline outputs;
read_track_figure draws them."""
import numpy as np
import plotly.graph_objects as go
import polars as pl

from kinnex_qc.patterns import MAS_ORDER
from kinnex_qc.scan import Hit

N_PER_STRATUM = 50
SEED = 20260924
TOL = 5

# Read features. Adapters A..H take eight categorical colours in order (checked as a set for colour-vision
# deficiencies); TSO, R1, polyT and cDNA use neutral steps. Every block is also labelled with its name, so identity
# never rests on colour alone. Q is a ninth adapter: its colour was checked pairwise against every one of A..H and
# against TSO by contrast, since a grey step for Q collides with TSO or R1 on the same ramp.
FEATURE_COLORS = {
    "A": "#2a78d6",
    "B": "#eb6834",
    "C": "#1baf7a",
    "D": "#eda100",
    "E": "#e87ba4",
    "F": "#008300",
    "G": "#4a3aa7",
    "H": "#e34948",
    "Q": "#e713b2",             # ninth hue, distinct from TSO and every adapter
    "TSO": "#0b0b0b",           # near black
    "R1": "#8a8985",            # mid neutral
    "polyT": "#c3c2b7",         # light neutral
    "cDNA": "#e4e3de",          # very light neutral
    "unannotated": "#ffffff",   # outline only
}

BEAD_BLOCK = 43          # CLS1 + GTGA + CLS2 + GACA + CLS3 + UMI after Read1
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# Track heights: adapters full height, primers lower, polyT lower still, so "adapter" vs
# "primer" reads from shape alone. cDNA and unannotated keep ADAPTER_W as the full-height base layer.
ADAPTER_W = 0.42
PRIMER_W = 0.24          # ~57% of ADAPTER_W (target 55-60%)
POLYT_W = 0.14
TRACK_WIDTHS = {"TSO": PRIMER_W, "R1": PRIMER_W, "polyT": POLYT_W}

# Per-read row geometry: adapter letters sit above the bar top (0.21 above row centre). Below the bar,
# top to bottom, sit the skera boundary ticks, then the class-label bracket with its label centred on the
# line (splitting it into two segments around the text). Every element's vertical extent (bar half-height,
# or text/marker size turned into row units via ROW_PX, the target pixels per row set via the figure height
# below) is checked by tests/test_read_tracks.py::test_row_layout_has_min_gap so that, row over row, staggered
# letter, letter, bar, tick, bracket+label and the next row's staggered letter never overlap and always
# clear each other by at least 0.03 row units.
LETTER_DY = 0.31         # adapter letter, above the bar
TICK_DY = 0.28           # skera S-read boundary tick, below the bar
BRACKET_DY = 0.40        # class-label bracket, and its centred label, below the tick
LETTER_STAGGER = 0.15    # extra lift for a letter whose neighbour on the same row sits too close
LETTER_FONT_SIZE = 13    # adapter letter font, readable at 50 reads on a wide render
LABEL_FONT_SIZE = 11     # class-label font
TICK_MARKER_SIZE = 9     # skera boundary tick marker, px
NOMINAL_PLOT_PX = 1370   # plot area width at the 1800 px export width (minus the l=150, r=280 margins)
LABEL_CHAR_PX = 0.62     # bold label character width, as a fraction of the font size
LABEL_PAD_PX = 12        # clear space kept on each side of a class label
ROW_PX = 150

# Legend order: adapters, then primers, then insert, then unannotated, then skera, then hatch.
LEGEND_RANK = {letter: i for i, letter in enumerate(MAS_ORDER)}
LEGEND_RANK.update({"TSO": 10, "R1": 11, "polyT": 12, "cDNA": 13, "unannotated": 14})
SKERA_LEGEND_RANK, HATCH_LEGEND_RANK = 15, 16
LEGEND_GROUP_TITLE = {"TSO": "Primers", "cDNA": "Insert and marks"}
DISPLAY_NAME = {"TSO": "TSO primer", "R1": "Read1 + cell label/UMI"}

READ_CAPTION = ("each molecule reads Read1 ... insert ... TSO, with the TSO next to the higher-index "
                "adapter; a TSO-TSO molecule has TSO at both ends and sits between an adapter and its "
                "own reverse complement (X and X')")


def strata(summary: pl.DataFrame, segs: pl.DataFrame, features: pl.DataFrame, skera: pl.DataFrame) -> dict[str, list[int]]:
    """Stratum name -> sorted ZMW ids (all candidates, before sampling). HiFi reads only.

    complete_proper: complete chain accepted by skera, all 8 of our segments proper.
    tso_tso / r1_r1: reads with a non-internal segment of that class.
    skera_rejected_<pattern>: reads skera put in the non-passing BAM, by primary chain pattern.
    skera_missplit: skera-passing reads where one of our MAS hits lies inside a skera S-read.
    """
    summary = summary.filter(pl.col("source") == "hifi")
    segs = segs.filter(pl.col("source") == "hifi")
    out = {}
    complete = summary.filter((pl.col("primary_pattern") == "complete") & pl.col("skera_passing"))["zm"]
    # A semi join, not is_in: is_in against a multi-million Series costs n_chunks x len(series) on a frame
    # read one chunk per parquet row group (polars 1.5.0), and complete runs into the millions at full scale.
    proper8 = (segs.join(complete.to_frame(), on="zm", how="semi").group_by("zm")
               .agg((pl.col("end_class") == "proper").sum().alias("n")).filter(pl.col("n") == 8)["zm"])
    out["complete_proper"] = proper8.sort().to_list()
    for c in ("tso_tso", "r1_r1"):
        out[c] = segs.filter((pl.col("end_class") == c) & ~pl.col("internal"))["zm"].unique().sort().to_list()
    rej = summary.filter(~pl.col("skera_passing"))
    for (pattern,), grp in rej.group_by("primary_pattern"):
        out[f"skera_rejected_{pattern}"] = grp["zm"].sort().to_list()
    passing = summary.filter(pl.col("skera_passing"))["zm"]
    mas = (features.filter((pl.col("source") == "hifi") & (pl.col("kind") == "MAS"))
           .join(passing.to_frame(), on="zm", how="semi").select("zm", "start", "end").sort("zm", "start"))
    sk = skera.select("zm", pl.col("qs").alias("start"), "qs", "qe").sort("zm", "start")
    j = mas.join_asof(sk, on="start", by="zm", strategy="backward")
    inside = j.filter((pl.col("qs") + TOL < pl.col("start")) & (pl.col("end") < pl.col("qe") - TOL))
    out["skera_missplit"] = inside["zm"].unique().sort().to_list()
    return {k: v for k, v in sorted(out.items()) if v}


def sample(zms: list[int], n: int = N_PER_STRATUM, seed: int = SEED) -> list[int]:
    """Up to n ZMW ids, chosen with a fixed seed, returned sorted."""
    if len(zms) <= n:
        return sorted(zms)
    rng = np.random.default_rng(seed)
    return sorted(np.asarray(sorted(zms))[rng.choice(len(zms), n, replace=False)].tolist())


def tracks(zms: list[int], summary: pl.DataFrame, segs: pl.DataFrame, features: pl.DataFrame, skera: pl.DataFrame) -> list[dict]:
    """Track data for the given reads, in ZMW order: read length, our hits, our segments, skera S-read boundaries."""
    keep = pl.Series(zms, dtype=pl.UInt32)
    summ = summary.filter((pl.col("source") == "hifi") & pl.col("zm").is_in(keep))
    f = features.filter((pl.col("source") == "hifi") & pl.col("zm").is_in(keep)).partition_by("zm", as_dict=True)
    s = segs.filter((pl.col("source") == "hifi") & pl.col("zm").is_in(keep)).partition_by("zm", as_dict=True)
    k = skera.filter(pl.col("zm").is_in(keep)).partition_by("zm", as_dict=True)
    out = []
    for zm, read_len, chain, pattern in summ.select("zm", "read_len", "chain", "primary_pattern").sort("zm").iter_rows():
        hits = [Hit(*r) for r in f[(zm,)].select("name", "kind", "strand", "start", "end", "edits").iter_rows()] \
            if (zm,) in f else []
        seg_rows = s[(zm,)].sort("start").select("start", "end", "end_class", "polyT_len", "tso_side").to_dicts() if (zm,) in s else []
        sk = k[(zm,)].sort("qs").select("qs", "qe").rows() if (zm,) in k else []
        out.append({"zm": zm, "read_len": read_len, "chain": chain, "pattern": pattern,
                    "features": hits, "segments": seg_rows, "skera": sk})
    return out


def _bars(name, color, y, starts, lengths, width, hover, pattern=None, legend=True, offset=None,
          line_color=None, line_width=0):
    line = dict(width=line_width)
    if line_color:
        line["color"] = line_color
    return go.Bar(y=y, x=lengths, base=starts, orientation="h", width=width, offset=offset, name=name,
                  marker=dict(color=color, line=line, pattern=dict(shape=pattern, fgcolor=SURFACE) if pattern else None),
                  hovertext=hover, hoverinfo="text", showlegend=legend, legendgroup=name)


def _gaps(read_len: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Complement of the union of (start, end) spans in [0, read_len)."""
    out, cur = [], 0
    for a, b in sorted(spans):
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < read_len:
        out.append((cur, read_len))
    return out


def _label_bp(text: str, longest: int) -> float:
    """Estimated width in bp of a bold class label plus its padding, on an x axis spanning 1.01 x longest."""
    px = len(text) * LABEL_CHAR_PX * LABEL_FONT_SIZE + 2 * LABEL_PAD_PX
    return px * longest * 1.01 / NOMINAL_PLOT_PX


def _subtract(span: tuple[float, float], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Parts of span left after removing every cut interval."""
    out = [span]
    for lo, hi in cuts:
        out = [piece for a, b in out for piece in ((a, min(b, lo)), (max(a, hi), b)) if piece[1] > piece[0]]
    return out


def read_track_figure(reads: list[dict], title: str) -> go.Figure:
    """One horizontal track per read, to scale. reads: dicts with zm, read_len, features (list[Hit]),
    segments (dicts with start, end, end_class, polyT_len), optional skera ([(qs, qe)]) and pattern.

    MAS adapters (A..H, Q) are drawn full track height, coloured per letter (Q distinct from every other
    colour, including TSO), hatched when on the '-' strand, and carry their letter just above the track,
    clear of the bar (with a prime for the minus strand, e.g. B'); a letter whose neighbour on the same
    row sits too close is lifted a little higher so the two never merge. Primers are drawn shorter so "adapter"
    and "primer" read from shape alone: TSO and Read1 (drawn together with the 43 nt cell label and UMI
    block) at PRIMER_W, polyT lower still at POLYT_W. cDNA is drawn per segment span at full track height
    as the light base fill, so each stretch carries its segment's hover; a read stretch covered by neither
    a segment nor a feature is drawn outline-only as "unannotated". Non-proper segments (tso_tso, r1_r1,
    ...) get a bold label centred on a thin bracket line spanning the segment (the line split into two
    segments around the label, with a gap in between), both in the secondary ink colour, below the bar and
    clear of the adapter letters above it, so they read without hovering. Ticks just below the track mark
    skera S-read boundaries, clear of both the bar above and the class-label bracket below. Features
    narrower than 1/300 of the longest read are
    widened so they stay visible (hover gives the true coordinates). The legend renders as a single vertical
    column to the right of the tracks (grouped, in a fixed rank order), since a horizontal legend with group
    titles can wrap a group's title and its first entry onto their own line out of order.
    """
    fig = go.Figure()
    if not reads:
        fig.update_layout(title=title, height=200, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE)
        return fig
    longest = max(r["read_len"] for r in reads)
    min_w = longest / 300
    ys = list(range(len(reads)))
    per = {}
    lbl_x, lbl_y, lbl_text = [], [], []
    brk_x, brk_y = [], []
    for i, r in enumerate(reads):
        def add(key, a, b, text, pattern=None):
            d = per.setdefault(key, {"y": [], "s": [], "l": [], "h": [], "p": []})
            d["y"].append(i)
            d["s"].append(a)
            d["l"].append(max(b - a, min_w))
            d["h"].append(text)
            d["p"].append(pattern or "")
        for h in r["features"]:
            span = f"{h.name} {h.strand} {h.start:,}-{h.end:,}, {h.edits} edits"
            if h.kind == "MAS":
                add(("MAS", h.name), h.start, h.end, span, "/" if h.strand == "-" else None)
            elif h.kind == "TSO":
                seg = next((s for s in r["segments"] if s["start"] <= h.start and h.end <= s["end"]), None)
                add(("TSO",), h.start, h.end, span + (f", segment: {seg['end_class']}" if seg else ""))
            else:
                a, b = (h.start, h.end + BEAD_BLOCK) if h.strand == "+" else (h.start - BEAD_BLOCK, h.end)
                seg = next((s for s in r["segments"] if s["start"] <= h.start and h.end <= s["end"]), None)
                seg_txt = f", segment: {seg['end_class']}" if seg else ""
                add(("R1",), a, b, span + " (with cell label and UMI)" + seg_txt)
                if seg and seg.get("polyT_len"):
                    n = seg["polyT_len"]
                    a, b = (b, b + n) if h.strand == "+" else (a - n, a)
                    add(("polyT",), a, b, f"polyT {n} nt{seg_txt}")
        row_labels, row_brackets = [], []
        for s in r["segments"]:
            add(("cDNA",), s["start"], s["end"], f"cDNA {s['start']:,}-{s['end']:,}, segment: {s['end_class']}")
            if s["end_class"] != "proper":
                row_labels.append([(s["start"] + s["end"]) / 2, s["end_class"]])
                row_brackets.append((s["start"], s["end"]))
        # Labels whose estimated extents would touch are merged into one; each bracket then spans its
        # segment minus every label extent in the row, so no line runs through text on short segments.
        row_labels.sort()
        merged = []
        for x, text in row_labels:
            if merged and x - _label_bp(text, longest) / 2 < merged[-1][0] + _label_bp(merged[-1][1], longest) / 2:
                px, ptext = merged.pop()
                x, text = (px + x) / 2, f"{ptext}, {text}"
            merged.append([x, text])
        cuts = [(x - _label_bp(t, longest) / 2, x + _label_bp(t, longest) / 2) for x, t in merged]
        for x, text in merged:
            lbl_x.append(x)
            lbl_y.append(i + BRACKET_DY)
            lbl_text.append(text)
        for a, b in row_brackets:
            for lo, hi in _subtract((a, b), cuts):
                brk_x += [lo, hi, None]
                brk_y += [i + BRACKET_DY, i + BRACKET_DY, None]
        mas_spans = [(h.start, h.end) for h in r["features"] if h.kind == "MAS"]
        seg_spans = [(s["start"], s["end"]) for s in r["segments"]]
        for a, b in _gaps(r["read_len"], mas_spans + seg_spans):
            add(("unannotated",), a, b, f"unannotated {a:,}-{b:,}")
    order = [("cDNA",), ("unannotated",), ("TSO",), ("R1",), ("polyT",)] + [("MAS", x) for x in MAS_ORDER]
    # The "Adapters" title goes on whichever adapter is actually drawn first (lowest LEGEND_RANK): a TSO-TSO
    # read rarely has an A, and a title that only ever sits on "A" then never appears at all.
    first_adapter = next((x for x in MAS_ORDER if ("MAS", x) in per), None)
    adp_x, adp_y, adp_text = [], [], []
    for key in order:
        if key not in per:
            continue
        d = per[key]
        name = key[1] if key[0] == "MAS" else key[0]
        outline = dict(line_color=INK2, line_width=1) if name == "unannotated" else {}
        width = TRACK_WIDTHS.get(name, ADAPTER_W)
        bar = _bars(name, FEATURE_COLORS[name], d["y"], d["s"], d["l"], width, d["h"], legend=False, **outline)
        if any(d["p"]):
            bar.marker.pattern = dict(shape=d["p"], fgcolor=SURFACE, size=5)
        fig.add_trace(bar)
        if key[0] == "MAS":
            for row, start, length, pat in zip(d["y"], d["s"], d["l"], d["p"]):
                adp_x.append(start + length / 2)
                adp_y.append(row)
                adp_text.append(name + ("'" if pat == "/" else ""))
        marker = dict(size=10, symbol="square", color=FEATURE_COLORS[name])
        if name == "unannotated":
            marker["line"] = dict(color=INK2, width=1)
        title = "Adapters" if name == first_adapter else LEGEND_GROUP_TITLE.get(name)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=DISPLAY_NAME.get(name, name),
                                 legendgroup=name, legendgrouptitle_text=title,
                                 legendrank=LEGEND_RANK[name], marker=marker))
    if adp_x:
        # Letters that share a row are staggered when close in x, so two adjacent adapters at 50-read
        # scale never merge into one glyph: the closest-below-min-gap gets lifted an extra notch.
        min_gap = longest / 70
        by_row: dict[float, list[int]] = {}
        for idx, row in enumerate(adp_y):
            by_row.setdefault(row, []).append(idx)
        drop = set()
        for row, idxs in by_row.items():
            idxs.sort(key=lambda idx: adp_x[idx])
            last = [None, None]          # index of the last letter placed at each level (base, lifted)
            for idx in idxs:
                level = next((lv for lv in (0, 1) if last[lv] is None or adp_x[idx] - adp_x[last[lv]] >= min_gap), None)
                if level is None:        # both levels crowded: join the letter to the last lifted one
                    adp_text[last[1]] += " " + adp_text[idx]
                    drop.add(idx)
                    continue
                adp_y[idx] = row - LETTER_DY - LETTER_STAGGER * level
                last[level] = idx
        keep = [k for k in range(len(adp_x)) if k not in drop]
        adp_x, adp_y, adp_text = [adp_x[k] for k in keep], [adp_y[k] for k in keep], [adp_text[k] for k in keep]
        fig.add_trace(go.Scatter(x=adp_x, y=adp_y, mode="text", text=adp_text,
                                 textfont=dict(color=INK, size=LETTER_FONT_SIZE), cliponaxis=False,
                                 hoverinfo="skip", showlegend=False))
    if lbl_x:
        fig.add_trace(go.Scatter(x=lbl_x, y=lbl_y, mode="text", text=[f"<b>{t}</b>" for t in lbl_text],
                                 textfont=dict(color=INK2, size=LABEL_FONT_SIZE), cliponaxis=False,
                                 hoverinfo="skip", showlegend=False))
    if brk_x:
        fig.add_trace(go.Scatter(x=brk_x, y=brk_y, mode="lines", line=dict(color=INK2, width=1.5),
                                 hoverinfo="skip", showlegend=False))
    sk_x, sk_y, sk_h = [], [], []
    for i, r in enumerate(reads):
        for qs, qe in r.get("skera", []):
            for x in (qs, qe):
                sk_x.append(x)
                sk_y.append(i + TICK_DY)
                sk_h.append(f"skera S-read {qs:,}-{qe:,}")
    if any(h.kind == "MAS" and h.strand == "-" for r in reads for h in r["features"]):
        fig.add_trace(go.Bar(x=[None], y=[None], name="hatched: adapter on the minus strand", legendrank=HATCH_LEGEND_RANK,
                             marker=dict(color="#8a8985", pattern=dict(shape="/", fgcolor=SURFACE))))
    if sk_x:
        fig.add_trace(go.Scatter(x=sk_x, y=sk_y, mode="markers", name="skera S-read boundary", legendrank=SKERA_LEGEND_RANK,
                                 hovertext=sk_h, hoverinfo="text",
                                 marker=dict(symbol="line-ns-open", size=TICK_MARKER_SIZE, color=INK2, line=dict(width=1.5))))
    # Vertical legend, fixed to the right: a horizontal legend with group titles can wrap a group's title
    # and its first entry onto their own line, breaking the rank order. A single column has no
    # wrap point, so LEGEND_RANK always renders top to bottom in the order it defines.
    fig.update_layout(title=dict(text=title, font=dict(color=INK, size=16),
                                 subtitle=dict(text=READ_CAPTION, font=dict(color=INK2, size=11))),
                      barmode="overlay", bargap=0,
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK2, size=11),
                      width=1800, height=max(950, 150 + ROW_PX * (len(reads) + 0.3)), margin=dict(l=150, r=280, t=90, b=60),
                      hovermode="closest",
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top", font=dict(size=10)))
    fig.update_yaxes(tickvals=ys, ticktext=[str(r["zm"]) for r in reads], range=[len(reads) - 0.4, -0.7], title="ZMW", showgrid=False)
    fig.update_xaxes(title="position in HiFi read (bp)", showgrid=True, gridcolor="#e4e3de", range=[0, longest * 1.01])
    return fig
