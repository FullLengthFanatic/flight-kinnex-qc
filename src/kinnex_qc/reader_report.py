"""report.html: what `kinnex-qc run` found, written for a reader who did not follow the analysis behind it.

Built from summary.json's numbers and a few example reads (pick_examples). The page is self-contained: plotly is
inlined once, fonts are system fonts, nothing is loaded from the network.
"""
import html
import importlib.metadata
import json
import platform
from importlib.resources import files
from pathlib import Path
from string import Template

import numpy as np
import plotly
import plotly.graph_objects as go
import polars as pl

from . import skera
from .sampling import hash_key_array
from .schematics import FATE, false_cut_svg
from .viewer import CSS as VIEWER_CSS, html_sequence

FONT = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
INK, INK2, RULE, GRID = "#1B1F24", "#5B636E", "#C9D0D7", "#EDF0F3"
NEUTRAL = ["#E2E7EC", "#A7B1BB", "#5E6A76", "#2B333B"]       # light to dark, one cool-biased neutral ramp
TSO_TSO_COLOUR = "#D55E00"

CATEGORIES = ("complete", "tso_tso", "r1_r1", "false_cut")
TITLE = {"complete": "Complete array", "tso_tso": "TSO-TSO mirror", "r1_r1": "R1-R1 mirror", "false_cut": "skera false cut"}
KIND = {"MAS": "adapter", "TSO": "tso", "R1": "r1"}
CLASS_NAME = {"proper": "proper", "tso_tso": "TSO-TSO", "r1_r1": "R1-R1", "tso_only": "TSO only", "r1_only": "Read1 only",
              "none": "no primer", "other": "other"}
_CHUNK = 500
WRAP = 100                    # bases per line of an example read


# ---------------------------------------------------------------- example reads

def clean(read_summary: pl.DataFrame, molecules: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """The reads clean enough to show, with their columns from read_summary: no fused molecule, every adapter an
    exact match, MAS hits = molecules + 1 and primer hits = 2 x molecules (molecules: the n_mas - 1 stretches
    between adapters), every primer within 1 edit, and rq >= 0.999."""
    mols = molecules.group_by("zm").agg(pl.col("internal").any().alias("fused"))
    fsum = features.group_by("zm").agg(
        (pl.col("kind") == "MAS").sum().cast(pl.Int64).alias("n_mas_hits"),
        pl.col("edits").filter(pl.col("kind") == "MAS").max().alias("mas_edits"),
        (pl.col("kind") != "MAS").sum().cast(pl.Int64).alias("n_primers"),
        pl.col("edits").filter(pl.col("kind") != "MAS").max().alias("primer_edits"))
    n_mol = pl.col("n_mas").cast(pl.Int64) - 1
    return (read_summary.join(mols, on="zm").join(fsum, on="zm")
            .filter(~pl.col("fused") & (pl.col("mas_edits") == 0) & (pl.col("n_mas_hits") == n_mol + 1)
                    & (pl.col("n_primers") == 2 * n_mol) & (pl.col("primer_edits") <= 1) & (pl.col("rq") >= 0.999)))


def _zm_series(zms) -> pl.Series:
    return pl.Series("zm", list(zms), dtype=pl.UInt32)


def at_adapters(block: list[int], sreads: pl.DataFrame, features: pl.DataFrame, tol: int = skera.TOL) -> set[int]:
    """The ZMWs of block whose every S-read starts within tol bp of the end of one of our adapters and ends within
    tol bp of the start of one."""
    s = (sreads.filter(pl.col("zm").is_in(_zm_series(block))).with_row_index("sid")
         .select("sid", "zm", pl.col("qs").cast(pl.Int64), pl.col("qe").cast(pl.Int64)))
    mas = (features.filter((pl.col("kind") == "MAS") & pl.col("zm").is_in(_zm_series(block)))
           .select("zm", pl.col("start").cast(pl.Int64), pl.col("end").cast(pl.Int64)))
    per = s.join(mas, on="zm").group_by("sid", "zm").agg(
        ((pl.col("qs") - pl.col("end")).abs() <= tol).any().alias("starts_ok"),
        ((pl.col("qe") - pl.col("start")).abs() <= tol).any().alias("ends_ok"))
    bad = set(per.filter(~(pl.col("starts_ok") & pl.col("ends_ok")))["zm"].to_list())
    return set(per["zm"].to_list()) - bad


def _first_by_hash(zms: pl.Series, n: int, ok=None) -> list[int]:
    """The first n of the ZMWs in zms (repeats allowed) by hash_key, smallest first; with ok (a function of a list of
    ZMWs returning the set that qualifies), the first n that qualify, checked in blocks of _CHUNK so a large candidate
    set is never joined whole. The candidates, up to most of a cell's reads, are ordered as a numpy array (hash_key_array)
    rather than as Python ints, which take about 130 bytes each in a list, a set and a sort key."""
    candidates = zms.unique().to_numpy()
    ordered = candidates[np.argsort(hash_key_array(candidates))]
    if ok is None:
        return ordered[:n].tolist()
    out = []
    for i in range(0, len(ordered), _CHUNK):
        block = ordered[i:i + _CHUNK].tolist()
        good = ok(block)
        out += [z for z in block if z in good][:n - len(out)]
        if len(out) >= n:
            break
    return out


def chain_as_read(between: pl.DataFrame) -> str:
    rows = between.sort("start").select("end_class", "left_token", "right_token").rows()
    if not rows:
        return "no molecule between two adapters"
    out = [rows[0][1]]
    for cls, _, right in rows:
        out += ([{"tso_tso": "[TSO-TSO]", "r1_r1": "[R1-R1]"}[cls]] if cls in ("tso_tso", "r1_r1") else []) + [right]
    return " ".join(out)


def _classes(between: pl.DataFrame) -> str:
    counts = between.group_by("end_class", maintain_order=True).len().rows()
    return ", ".join(f"{k} {CLASS_NAME.get(c, c)}" for c, k in counts)


def _sread_text(rows, order) -> str:
    return "; ".join(f"{qs + 1:,}-{qe:,} ({order[dl]} to {order[dr]})" for qs, qe, dl, dr in rows)


def _example(category: str, zm: int, read_summary, molecules, features, order, sreads=None, cut=None) -> dict:
    """One example read's heading data, its note (what to look at) and its features in read coordinates."""
    mols = molecules.filter(pl.col("zm") == zm).sort("start")
    feats = features.filter(pl.col("zm") == zm).sort("start")
    read_len = int(read_summary.filter(pl.col("zm") == zm)["read_len"][0])
    marks = [(n + ("'" if s == "-" else ""), a, b, KIND[k]) for n, k, s, a, b in feats.select("name", "kind", "strand", "start", "end").iter_rows()]
    sr = (sreads.filter(pl.col("zm") == zm).sort("qs").select("qs", "qe", "dl", "dr").rows() if sreads is not None else None)
    between = mols.filter(pl.col("left_token").is_not_null() & pl.col("right_token").is_not_null())
    chain, classes = chain_as_read(between), _classes(between)
    if category == "complete":
        all_proper = (between["end_class"] == "proper").all()
        note = (f"As read: {chain}. {between.height} molecules between adapters, "
                + ("all proper (Read1 at one end, the TSO at the other)." if all_proper else f"{classes}."))
        if sr is not None:
            note += f" skera wrote {len(sr)} S-reads, one per molecule, each from one of our adapters to the next."
        focus = None
    elif category in ("tso_tso", "r1_r1"):
        j = skera.mirror_junctions(mols).row(0, named=True)
        primer = "a TSO" if category == "tso_tso" else "Read1"
        letter = j["left_token"].rstrip("'")
        note = (f"As read: {chain}. The molecule at {j['start'] + 1:,}-{j['end']:,} has {primer} at both ends and adapter {letter} "
                f"on both sides, the mirror junction. Molecules between adapters: {classes}.")
        if sr is not None:
            note += (f" skera wrote {len(sr)} S-read{'s' if len(sr) != 1 else ''}: {_sread_text(sr, order)}." if sr
                     else " skera wrote no S-read for this read.")
        focus = j["start"]
    else:
        called = order[cut["dr_before"]]
        inside = "before" if cut["in_sread"] == 1 else "after"
        mid = (cut["cut_start"] + cut["cut_end"]) // 2
        miss = min((m for m in marks if m[3] == "adapter" and m[0].rstrip("'") == cut["missed"]), key=lambda m: (abs(m[1] - mid), m[1]))
        note = (f"skera cut at {cut['cut_start'] + 1:,}-{cut['cut_end']:,} and called adapter {called} there, but none of our adapters lies "
                f"within {skera.TOL} bp of that stretch: a false cut. Our adapter {miss[0]} at {miss[1] + 1:,}-{miss[2]:,}, "
                f"{abs(miss[1] - mid):,} bp from the cut, lies inside the S-read {inside} the cut: skera skipped it. As read: {chain}.")
        marks.append((called, cut["cut_start"], cut["cut_end"], "false"))
        focus = cut["cut_start"]
    return {"category": category, "zm": zm, "read_len": read_len, "note": note, "features": marks, "focus": focus}


def pick_examples(read_summary: pl.DataFrame, molecules: pl.DataFrame, features: pl.DataFrame, n: int, *, order,
                  sreads: pl.DataFrame | None = None, passing: pl.DataFrame | None = None,
                  fc: pl.DataFrame | None = None) -> list[dict]:
    """Up to n clean example reads (see clean) per category, in CATEGORIES order and, within one, by hash_key, the
    order the hash sample admits ZMWs in, so the picks spread over the cell whatever the file order.

    complete: complete pattern; with sreads, also one S-read per molecule, each starting and ending at our
    adapters (within skera.TOL bp). tso_tso: one mirror junction, holding a TSO-TSO molecule. r1_r1: one mirror
    junction, holding an R1-R1 molecule. false_cut (only with sreads; passing and fc are skera.passing_from_labels'
    and skera.false_cuts' output): complete, passing, exactly one false cut, its skipped adapter found
    (skera.skipped_adapters), and a gap between its two S-reads (cut_end > cut_start; S-reads that touch or
    overlap leave no site to mark). Each example: category, zm, read_len, note (one line of what to look at) and features
    (name, start, end, kind) for viewer.html_sequence, and focus (the junction molecule's or the false cut's start, where
    the read's box opens; None for complete arrays); the caller adds the read's sequence as "seq"."""
    if n <= 0:
        return []
    clean_reads = clean(read_summary, molecules, features)
    junctions = skera.mirror_junctions(molecules).group_by("zm").agg(pl.len().alias("nj"), pl.col("end_class").first().alias("jclass"))
    single = junctions.filter(pl.col("nj") == 1)
    clean_zm = clean_reads["zm"]
    complete = clean_reads.filter(pl.col("primary_pattern") == "complete")
    picks = {"tso_tso": _first_by_hash(skera.single_tso_tso_mirrors(molecules).filter(pl.col("zm").is_in(clean_zm))["zm"], n),
             "r1_r1": _first_by_hash(single.filter((pl.col("jclass") == "r1_r1") & pl.col("zm").is_in(clean_zm))["zm"], n)}
    cuts = {}
    if sreads is None:
        picks["complete"] = _first_by_hash(complete["zm"], n)
    else:
        n_sreads = sreads.group_by("zm").len().rename({"len": "n_sreads"}).with_columns(pl.col("n_sreads").cast(pl.Int64))
        one_each = complete.join(n_sreads, on="zm").filter(pl.col("n_sreads") == pl.col("n_mas").cast(pl.Int64) - 1)["zm"]
        picks["complete"] = _first_by_hash(one_each, n, ok=lambda block: at_adapters(block, sreads, features))
        target = complete.join(passing.filter(pl.col("passing")), on="zm").select("zm")
        fc_t = fc.join(target, on="zm")
        one = fc_t.filter(pl.len().over("zm") == 1)
        s, f = sreads.filter(pl.col("zm").is_in(one["zm"])), features.filter(pl.col("zm").is_in(one["zm"]))
        found = (skera.skipped_adapters(one, s, f).filter(pl.col("missed").is_not_null())
                 .join(one.select("zm", "cut_start", "cut_end", pl.col("dr_before").cast(pl.Int64)), on=["zm", "cut_start"])
                 .filter(pl.col("cut_end") > pl.col("cut_start")))
        picks["false_cut"] = _first_by_hash(found["zm"], n)
        cuts = {r["zm"]: r for r in found.filter(pl.col("zm").is_in(_zm_series(picks["false_cut"]))).iter_rows(named=True)}
    return [_example(c, zm, read_summary, molecules, features, order, sreads=sreads, cut=cuts.get(zm))
            for c in CATEGORIES if c in picks for zm in picks[c]]


# ---------------------------------------------------------------- formatting

def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _pct(v, digits: int = 1) -> str:
    """A share as a percentage, fixed point: to `digits` decimals from 1%, to two below it (0.42%), "< 0.01%" below
    0.01% and "0%" for exactly zero."""
    if v is None:
        return "n/a"
    if v == 0:
        return "0%"
    if v < 0.0001:
        return "< 0.01%"
    if v < 0.01:
        return f"{100 * v:.2f}%"
    return f"{100 * v:.{digits}f}%"


def _num(v, digits: int = 2) -> str:
    """A count with thousands separators; a mean fixed point to `digits` decimals, "< 0.01" (for digits 2) when
    smaller than its last decimal and "0" for exactly zero."""
    if v is None:
        return "n/a"
    if isinstance(v, int):
        return f"{v:,}"
    if v == 0:
        return "0"
    floor = f"{10 ** -digits:.{digits}f}"
    if abs(v) < 10 ** -digits:
        return f"< {floor}" if v > 0 else f"> -{floor}"
    return f"{v:,.{digits}f}"


def _per1000(v) -> str:
    """A rate per 1,000, fixed point: one decimal from 10, two below, "< 0.01" below 0.01 and "0" for exactly zero."""
    if v is None:
        return "n/a"
    if v == 0:
        return "0"
    if v < 0.01:
        return "< 0.01"
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def _key(path: str, value) -> str:
    return f'<code class="key">{_e(path)} = {_e(json.dumps(value))}</code>'


def _table(header: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def _numbers(rows: list[tuple[str, str, str]]) -> str:
    """Each number with the plain sentence of what it counts and where summary.json holds it."""
    return _table(["Number", "What it counts", "In summary.json"], [[f'<span class="val">{v}</span>', s, k] for v, s, k in rows], "numbers")


def _letters(order) -> str:
    order = list(order)
    return ", ".join(order[:-1]) + " and " + order[-1] if len(order) > 1 else order[0]


# ---------------------------------------------------------------- figures

def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, font=dict(family=FONT, size=12, color=INK2),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", margin=dict(l=150, r=30, t=20, b=50),
                      legend=dict(font=dict(size=11, color=INK2), bgcolor="rgba(255,255,255,0)", traceorder="normal"),
                      hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor="#FFFFFF", bordercolor=RULE))
    fig.update_xaxes(showline=True, linecolor=INK2, linewidth=1, ticks="outside", tickcolor=INK2, gridcolor=GRID, zeroline=False,
                     title_font=dict(size=12, color=INK2))
    fig.update_yaxes(showline=True, linecolor=INK2, linewidth=1, ticks="outside", tickcolor=INK2, gridcolor=GRID, zeroline=False,
                     title_font=dict(size=12, color=INK2))
    return fig


def _plot(fig: go.Figure) -> str:
    div = plotly.offline.plot(fig, include_plotlyjs=False, output_type="div",
                              config={"displayModeBar": False, "responsive": True})
    return f'<div class="plotwrap">{div}</div>'


PATTERNS = [("complete", "Complete", NEUTRAL[0]), ("truncated", "Truncated", NEUTRAL[1]), ("other", "Other", NEUTRAL[2]),
            ("mirror", "Mirror junction", NEUTRAL[3])]


def _pattern_figure(shares: dict) -> str:
    f = go.Figure()
    for key, lab, col in PATTERNS:
        v = 100 * shares.get(key, 0.0)
        f.add_trace(go.Bar(y=[lab], x=[v], orientation="h", name=lab, showlegend=False,
                           marker=dict(color=col, line=dict(color=NEUTRAL[1], width=1)),
                           text=[_pct(v / 100)], textposition="outside", textfont=dict(size=12, color=INK), cliponaxis=False,
                           hovertemplate=f"{lab}: %{{x:.2f}}% of HiFi reads<extra></extra>"))
    f.update_xaxes(range=[0, 112], tickvals=[0, 20, 40, 60, 80, 100], title_text="HiFi reads (%)")
    return _plot(_style(f, 230).update_yaxes(autorange="reversed", showline=False, ticks=""))


def _junction_figure(junctions: list[dict], order) -> str:
    f = go.Figure()
    for mol, col in (("TSO-TSO", TSO_TSO_COLOUR), ("R1-R1", NEUTRAL[2]), ("other", NEUTRAL[1])):
        sub = {r["letter"]: r["junctions"] for r in junctions if r["molecule"] == mol}
        f.add_trace(go.Bar(x=list(order), y=[sub.get(x, 0) for x in order], name=f"{mol} molecule at the junction",
                           marker=dict(color=col, line=dict(color="#FFFFFF", width=1)),
                           hovertemplate="adapter %{x}: %{y:,} junctions<extra>" + mol + "</extra>"))
    f.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.3, x=0))
    f.update_xaxes(title_text="adapter on both sides of the junction", type="category")
    top = max((sum(r["junctions"] for r in junctions if r["letter"] == x) for x in order), default=0)
    f.update_yaxes(title_text="mirror junctions", rangemode="tozero", dtick=1 if top <= 10 else None)
    return _plot(_style(f, 320).update_layout(margin=dict(l=70)))


def _budget_other(b: dict):
    """Molecules per read that are neither usable nor one of the four artifact kinds: molecules - usable - artifact
    (skera.budget keys), at least 0. summary.json has no key of its own for it."""
    return None if b["molecules"] is None else max(b["molecules"] - b["usable"] - b["artifact"], 0.0)


def _budget_figure(b: dict, slots: int) -> str:
    other = _budget_other(b)
    sep = dict(color="#FFFFFF", width=1)
    segs = [("Usable molecules skera writes cleanly", b["usable_clean"], dict(color=NEUTRAL[2], line=sep), "#FFFFFF"),
            ("Usable molecules skera does not write cleanly (a full-read segmentation keeps them)", b["usable_not_clean"],
             dict(color="#FFFFFF", line=sep, pattern=dict(shape="/", fgcolor=FATE["unwritten"], bgcolor="#FFFFFF", size=8, solidity=0.5)), INK),
            ("Artifact molecules (TSO-TSO, R1-R1, one primer or none, fused)", b["artifact"], dict(color=FATE["artifact"], line=sep), INK),
            ("Other molecules (primers in no expected arrangement)", other, dict(color=NEUTRAL[0], line=sep), INK),
            ("Slots the array does not fill", b["unfilled_slots"], dict(color=FATE["unfilled"], line=sep), INK)]
    f = go.Figure()
    for lab, v, marker, ink in segs:
        if v <= 0:
            continue
        f.add_trace(go.Bar(y=["per HiFi read"], x=[v], orientation="h", name=f"{lab}: {_num(v)}", marker=marker,
                           text=[f"{v:.2f}" if v >= 0.4 else ""], textposition="inside", insidetextanchor="middle", textangle=0,
                           textfont=dict(size=12, color=ink), hovertemplate=f"{_e(lab)}: %{{x:.3f}} per HiFi read<extra></extra>"))
    f.update_layout(barmode="stack", bargap=0.35, legend=dict(orientation="h", y=-0.45, x=0, traceorder="normal"))
    f.update_xaxes(title_text=f"molecules per HiFi read, out of {slots} slots", range=[0, max(slots, b["molecules"]) * 1.02], dtick=1)
    return _plot(_style(f, 300).update_layout(margin=dict(l=20, r=20, t=10, b=40)).update_yaxes(showticklabels=False, showline=False, ticks=""))


# ---------------------------------------------------------------- sections

def _header(s: dict) -> str:
    inp = s["inputs"]
    hifi = Path(inp["hifi"]).name
    sample = ("every ZMW in the file" if s["fraction"] >= 1 else
              f"a hash sample of {_pct(s['fraction'])} of the ZMWs, {s['hifi_reads_in_file']:,} HiFi reads in the file")
    sreads = (f" skera's S-reads: {_e(Path(inp['sreads']).name)} (skera {_e(s['skera_version'] or 'version not in the header')})."
              if inp["sreads"] else " No skera S-reads given (--sreads), so skera is not compared.")
    return (f'<div class="eyebrow">kinnex-qc {_e(s["version"])} report</div>'
            f'<h1>Kinnex array check of {_e(hifi)}</h1>'
            f'<p class="byline">{s["hifi_reads"]:,} HiFi reads checked ({sample}).{sreads}</p>')


def _toc(has_skera: bool) -> str:
    items = [("s1", "1 What this checks"), ("s2", "2 Array structure"), ("s3", "3 Artifact molecules")]
    if has_skera:
        items.append(("skera", "4 skera"))
    items += [("examples", f"{5 if has_skera else 4} Example reads"), ("inputs", f"{6 if has_skera else 5} Inputs and limits")]
    return '<nav class="toc" aria-label="Sections">' + "".join(f'<a href="#{i}">{t}</a>' for i, t in items) + "</nav>"


def _section_checks(s: dict) -> str:
    order = s["chemistry"]["order"]
    kind = "Kinnex 8-fold array" if s["chemistry"]["adapters"] == "Kinnex 8-fold default" else "Kinnex array"
    return f"""<h2 id="s1"><span class="sec">1</span>What this report checks</h2>
<div class="prose"><p>A {kind} is one HiFi read made of {len(order) - 1} cDNA molecules joined end to end by {len(order)} adapters,
{_e(_letters(order))}, always in that order. This report finds every adapter and both cDNA primers (TSO and Read1) along each read,
checks that the adapters run in order, and counts the reads and molecules that break that order or carry the wrong primers.
With skera's S-reads (<code>--sreads</code>) it also checks where skera cut each read against the adapters found here.</p></div>
<div class="terms"><dl>
<div><dt>HiFi read, ZMW</dt><dd>A ZMW is one well of the SMRT Cell, where one DNA molecule is read many times over; the consensus of
those passes is one HiFi read. Here each HiFi read is one array.</dd></div>
<div><dt>Molecule</dt><dd>The stretch of read between two adapters (and a stretch of at least 50 bp before the first or after the last).
Each is classed by the primers in its first and last 150 bp.</dd></div>
<div><dt>Proper (usable) molecule</dt><dd>A cDNA molecule with the template-switching oligo (TSO) at one end and Read1, followed by the
cell label and UMI, at the other. Only these can be assigned to a cell.</dd></div>
<div><dt>TSO-TSO and R1-R1 molecules</dt><dd>Artifacts with a TSO (or Read1) at both ends. A TSO-TSO molecule carries no cell label.</dd></div>
<div><dt>Mirror junction</dt><dd>An adapter followed by its own reverse complement, for example H then H' (a prime marks a sequence found
on the opposite strand). A TSO-TSO or R1-R1 molecule takes the same adapter at both ends, so the read grows the same way on both sides
of it and mirrors itself around that molecule: a TSO-TSO or R1-R1 mirror.</dd></div>
<div><dt>skera, S-read, false cut</dt><dd>skera is PacBio's tool that splits Kinnex HiFi reads at adapter positions into segmented
reads, S-reads (documented at <a href="https://skera.how">skera.how</a>). A false cut is a place where skera cut although none of our
adapters is there (section 4).</dd></div>
</dl></div>"""


PATTERN_TEXT = {
    "complete": "every adapter once, in order, from the first to the last",
    "truncated": "adapters in order with none skipped, but the chain stops short of the first adapter, the last or both",
    "mirror": "at least one mirror junction: an adapter followed by its own reverse complement",
    "other": "a skipped slot (an adapter missing inside the chain), a repeated adapter, or no chain this check can place",
}


def _section_structure(s: dict) -> str:
    n = s["hifi_reads"]
    pats = {r["pattern"]: r for r in s["patterns"]}
    shares = {k: r["share"] for k, r in pats.items()}
    rows = [[lab, f"{pats.get(k, {'reads': 0})['reads']:,}", _pct(shares.get(k, 0.0)), PATTERN_TEXT[k]] for k, lab, _ in PATTERNS]
    mj = s["mirror_junction_shares"]
    order = s["chemistry"]["order"]
    out = [f"""<h2 id="s2"><span class="sec">2</span>Array structure</h2>
<div class="prose"><p>Each HiFi read gets one pattern from its chain of adapters. Of the {n:,} HiFi reads checked,
{_pct(shares.get('complete', 0.0) if n else None)} are complete arrays and {_pct(shares.get('mirror', 0.0) if n else None)} have a mirror junction.</p></div>
{_pattern_figure(shares) if n else ''}
<p class="legend">Share of the {n:,} HiFi reads checked, by the pattern of their adapter chain (summary.json: <code>patterns</code>).</p>
{_table(['Pattern', 'HiFi reads', 'Share of HiFi reads', 'What it means'], rows, 'num')}
<h3>Mirror junctions</h3>"""]
    if mj["mirror_junctions"]:
        out.append(f"""<div class="prose"><p>The {n:,} reads hold {mj['mirror_junctions']:,} mirror junctions. {_pct(mj['mirror_junction_tso_tso_share'])}
of them have a TSO-TSO molecule between the two adapters and {_pct(mj['mirror_junction_r1_r1_share'])} an R1-R1 molecule.</p></div>
{_junction_figure(s['junctions'], order)}
<p class="legend">Mirror junctions by the adapter on both sides and the molecule between them (summary.json: <code>junctions</code>).
Junction molecules that also hold a primer inside them are counted.</p>""")
    else:
        out.append('<div class="prose"><p>None of the reads checked has a mirror junction.</p></div>')
    out.append(_numbers([
        (f"{mj['mirror_junctions']:,}", "Mirror junctions in all reads checked: molecules with the same adapter letter on both sides, "
         "one forward and one reverse complement.", _key("mirror_junction_shares.mirror_junctions", mj["mirror_junctions"])),
        (_pct(mj["mirror_junction_tso_tso_share"]), "Share of those junctions whose molecule is TSO-TSO.",
         _key("mirror_junction_shares.mirror_junction_tso_tso_share", mj["mirror_junction_tso_tso_share"])),
        (_pct(mj["mirror_junction_r1_r1_share"]), "Share of those junctions whose molecule is R1-R1.",
         _key("mirror_junction_shares.mirror_junction_r1_r1_share", mj["mirror_junction_r1_r1_share"])),
    ]))
    if s["mirror_size_rule"]:
        rows = [[_e(r["letter"]), f"{r['reads']:,}", f"{r['predicted']}", _num(r["median_molecules"], 1), _pct(r["share_matching"])]
                for r in s["mirror_size_rule"]]
        out.append(f"""<div class="prose"><p>A TSO-TSO molecule takes the adapter of the TSO end of its slot at both ends, and the array
then grows from it towards the last adapter on both sides. A whole two-armed mirror with its junction at adapter X therefore holds
2 x (slots from X to the last adapter) + 1 molecules. The table compares that with the reads that have exactly one mirror junction,
holding a TSO-TSO molecule.</p></div>
{_table(['Junction adapter', 'Reads', 'Molecules predicted', 'Median molecules', 'Share with the predicted number'], rows, 'num')}
<p class="legend">summary.json: <code>mirror_size_rule</code>.</p>""")
    if s["junction_composition"]:
        rows = [[_e(r["junction_molecules"]), f"{r['reads']:,}", _num(r["median_molecules"], 1),
                 _pct(r["skera_pass_share"]) if r["skera_pass_share"] is not None else "not measured (no --sreads)"]
                for r in s["junction_composition"]]
        out.append(f"""<div class="prose"><p>Every read with at least one mirror junction, by the molecules its junctions hold: how many
such reads, the median number of molecules in them and the share skera passes (its S-read labels run from the first adapter to the last).</p></div>
{_table(['Junction molecules', 'Reads', 'Median molecules', 'skera passes'], rows, 'num')}
<p class="legend">summary.json: <code>junction_composition</code>.</p>""")
    return "\n".join(out)


ARTIFACT_TEXT = [
    ("proper", "Proper", "TSO at one end and Read1 at the other: a usable molecule"),
    ("tso_tso", "TSO-TSO", "a TSO at both ends and no Read1, so no cell label"),
    ("r1_r1", "R1-R1", "Read1 at both ends and no TSO"),
    ("tso_only", "TSO only", "a TSO at one end and no primer at the other"),
    ("r1_only", "Read1 only", "Read1 at one end and no primer at the other"),
    ("none", "No primer", "no primer in either end window"),
    ("other", "Other", "primers in no expected arrangement: one on the wrong strand, or two at one end"),
    ("internal", "Fused", "a TSO or Read1 inside the molecule: two molecules joined with no adapter between them, whatever their ends"),
]


def _section_artifacts(s: dict) -> str:
    a = s["artifacts"]
    rows = [[lab, _per1000(a[f"{k}_per_1000"]), text, _key(f"artifacts.{k}_per_1000", a[f"{k}_per_1000"])] for k, lab, text in ARTIFACT_TEXT]
    return f"""<h2 id="s3"><span class="sec">3</span>Artifact molecules per 1,000</h2>
<div class="prose"><p>Every molecule found in the {s['hifi_reads']:,} reads is classed by the primers at its two ends. Per 1,000 molecules,
{_per1000(a['tso_tso_per_1000'])} are TSO-TSO and {_per1000(a['r1_r1_per_1000'])} are R1-R1. The classes by their ends count only molecules
with no primer inside them; fused molecules are counted apart, whatever their ends.</p></div>
{_table(['Class', 'Per 1,000 molecules', 'What it is', 'In summary.json'], rows, 'num')}"""


def _section_skera(s: dict) -> str:
    k = s["skera"]
    order = s["chemistry"]["order"]
    letters = sorted(set(k["false_cut_letters"]) | set(k["skipped_adapter_letters"]),
                     key=lambda x: (x not in order, order.index(x) if x in order else 0, x))
    letter_rows = [[_e(x if x != "none" else "none found"), f"{k['false_cut_letters'].get(x, 0):,}", f"{k['skipped_adapter_letters'].get(x, 0):,}"]
                   for x in letters]
    ma = k["mirror_arms"]
    arm_rows = []
    for region, lab in (("after", "After the junction (forward adapters)"), ("before", "Before the junction (reverse-complement adapters)")):
        c = ma[f"{region}_counts"]
        arm_rows.append([lab, f"{c['CLEAN']:,}", f"{c['FUSED']:,}", f"{c['PARTIAL']:,}", f"{c['NOT_WRITTEN']:,}", _pct(ma[f"{region}_clean_share"])])
    n_after, n_before = sum(ma["after_counts"].values()), sum(ma["before_counts"].values())
    b = k["budget"]
    slots = len(order) - 1
    budget_rows = [
        ("molecules", "Molecules our segmentation finds, per HiFi read."),
        ("unfilled_slots", f"Slots the array does not fill: {slots} minus the molecules, per read (negative for a read with more molecules than slots, as a long mirror array has)."),
        ("artifact_tso_tso", "TSO-TSO molecules per read."),
        ("artifact_r1_r1", "R1-R1 molecules per read."),
        ("artifact_one_primer_or_none", "Molecules with one primer or none, per read."),
        ("artifact_fused", "Fused molecules (proper ends, a primer inside), per read."),
        ("artifact", "All artifact molecules per read: the four kinds above."),
        ("usable", "Usable molecules (proper, not fused) per read."),
        ("usable_clean", "Usable molecules skera writes cleanly: one S-read covers at least 80% of the molecule and at least 20% of no other."),
        ("usable_not_clean", "Usable molecules skera does not write cleanly: fused with a neighbour in one S-read, partly covered, or not written at all."),
    ]
    sk_version = _e(s["skera_version"] or "not named in the S-read BAM's header")
    return f"""<h2 id="skera"><span class="sec">4</span>skera's cuts against the adapters found here</h2>
<div class="prose"><p>skera version: {sk_version}. Here a read passes skera when the adapter labels of its S-reads run from the first
adapter to the last, in order; this command reads only the S-reads, not skera's non_passing.bam. skera passes {k['complete_passing_reads']:,} of the
{k['complete_reads']:,} complete arrays.</p></div>
<h3>False cuts</h3>
<div class="prose"><p>A false cut is a place where skera cut a complete array although none of our adapters lies within {skera.TOL} bp of
the cut. In the three BD144 libraries of the study behind this package (observed with skera 26.1.0 and lima 2.13.0), skera had skipped
a real adapter and cut at a similar sequence inside the next molecule, a median of about 900 bp past it. skera labelled the cut D in 74%
of false cuts in each library. The adapter it skipped, found at 99.6% of false cuts, was D in 74 to 75% of those and had the letter skera
gave the cut in 99%, so the labels still ran in order and the read still passed. Of the two S-reads around such a cut, lima passed the
chimera (the molecule before the skipped adapter, the adapter and the start of the next molecule) in under 0.5% of cases and the fragment
(the rest of that molecule) almost never. Our segmentation cuts at the real adapters and keeps both molecules; this command counts false
cuts and does not run lima.</p>
<p class="headline"><span class="big">{_pct(k['false_cut_reads_share'])}</span> of the {k['complete_passing_reads']:,} complete arrays
skera passes have at least one false cut; they hold {k['false_cuts']:,} false cut{'s' if k['false_cuts'] != 1 else ''} in all.</p></div>
<figure>{false_cut_svg(k)}<figcaption class="legend">One false cut, drawn to explain and not to scale, with the Kinnex 8-fold
letters and D as the skipped adapter. The cut position (at a D-like sequence inside the next molecule) and the strand rule (on reads from
the other strand the fragment keeps the Read1 end) are as observed in the three BD144 libraries (skera 26.1.0), not documented skera
behaviour. The lima verdicts of the study's drawing are left out: this command does not run lima.</figcaption></figure>
{_numbers([
    (f"{k['complete_reads']:,}", "Complete arrays among the reads checked.", _key("skera.complete_reads", k["complete_reads"])),
    (f"{k['complete_passing_reads']:,}", "Complete arrays that skera passes: the base of the false-cut numbers.",
     _key("skera.complete_passing_reads", k["complete_passing_reads"])),
    (f"{k['false_cuts']:,}", "False cuts in those reads: gaps between two consecutive S-reads with none of our adapters within "
     f"{skera.TOL} bp.", _key("skera.false_cuts", k["false_cuts"])),
    (_pct(k["false_cut_reads_share"]), "Share of the complete arrays skera passes with at least one false cut.",
     _key("skera.false_cut_reads_share", k["false_cut_reads_share"])),
    (_num(k["false_cuts_per_read"], 3), "False cuts per complete array skera passes.", _key("skera.false_cuts_per_read", k["false_cuts_per_read"])),
])}
{_table(['Adapter', 'False cuts skera called this adapter', 'False cuts that skipped our adapter of this letter'], letter_rows, 'num') if letter_rows else ''}
<p class="legend">The skipped adapter is the one of ours inside the S-read before or after the cut that lies closest to it; "none found" when
neither S-read holds one (summary.json: <code>skera.false_cut_letters</code>, <code>skera.skipped_adapter_letters</code>).</p>
<h3>Mirror arms</h3>
<div class="prose"><p>A read with one mirror junction holding a TSO-TSO molecule has two arms: the one after the junction, with forward
adapters (H to Q for a junction at H), and the one before it, with reverse-complement adapters (Q' to H'). In this run skera wrote
{_pct(ma['after_clean_share'])} of the {n_after:,} usable molecules on the arm after the junction cleanly, and {_pct(ma['before_clean_share'])}
of the {n_before:,} on the arm before it. In the BD144 ND library of the study behind this package skera 26.1.0 wrote 86% and 8% of
them cleanly; which arm skera writes is an observation on those reads, not documented skera behaviour.</p></div>
{_table(['Arm', 'Written cleanly', 'Fused with a neighbour', 'Partly covered', 'Not written', 'Share written cleanly'], arm_rows, 'num')}
<p class="legend">Usable molecules of reads with exactly one mirror junction, holding a TSO-TSO molecule. Written cleanly: one S-read covers at
least 80% of the molecule and at least 20% of no other. Fused with a neighbour: that S-read also covers at least 20% of another molecule.
Partly covered: the best S-read covers 20 to 80%. Not written: less (summary.json: <code>skera.mirror_arms</code>).</p>
<h3>Slot budget</h3>
<div class="prose"><p>A whole array has {slots} slots, one per molecule. Per HiFi read, on average, skera writes {_num(b['usable_clean'])} usable
molecules cleanly and leaves {_num(b['usable_not_clean'])} usable molecules not cleanly written; {_num(b['artifact'])} slots hold artifact
molecules and {_num(b['unfilled_slots'])} slots stay empty.</p></div>
{_budget_figure(b, slots) if b['molecules'] is not None else ''}
<p class="legend">Means over the {s['hifi_reads']:,} HiFi reads checked, reads without molecules counted as 0 (summary.json: <code>skera.budget</code>).
Other molecules, per read: molecules minus usable minus artifact molecules (<code>skera.budget</code> keys), that is molecules that are
neither usable nor one of the four artifact kinds; summary.json has no key of its own for them.</p>
{_numbers([(_num(b[c]), text, _key(f"skera.budget.{c}", b[c])) for c, text in budget_rows]
          + [(_num(_budget_other(b)), "Other molecules per read: molecules minus usable minus artifact molecules, the ones that are neither "
              "usable nor one of the four artifact kinds (primers in no expected arrangement).",
              'no key of its own: <code class="key">molecules - usable - artifact</code>')])}"""


def _section_examples(s: dict, examples: list[dict], n_section: int) -> str:
    has_sreads = "skera" in s
    kinds = [("Complete array", "a complete chain" + ("; skera wrote one S-read per molecule, each starting and ending at our adapters"
                                                      if has_sreads else "")),
             ("TSO-TSO mirror", "exactly one mirror junction, holding a TSO-TSO molecule"),
             ("R1-R1 mirror", "exactly one mirror junction, holding an R1-R1 molecule")]
    if has_sreads:
        kinds.append(("skera false cut", "a complete array skera passes, with exactly one false cut whose skipped adapter is found "
                      "and whose two S-reads leave a gap at the cut"))
    out = [f"""<h2 id="examples"><span class="sec">{n_section}</span>Example reads</h2>
<div class="prose"><p>Up to <code>--examples</code> reads (default 2) of each kind below, the first in hash order (the order the hash
sample admits ZMWs in, spread over the cell whatever the file order) among the clean reads: no fused molecule, every adapter an exact
match, two primer hits per molecule with at most one edit each, and read quality rq of at least 0.999.</p>
<ul>{"".join(f"<li><b>{k}</b>: {d}.</li>" for k, d in kinds)}</ul>
<p>Each read is shown whole. Positions are 1-based on the read; hover over a coloured stretch for its name and coordinates. A prime
(H', TSO') marks a sequence found as its reverse complement on the read. The box of a mirror or a false cut opens at the junction or the cut.</p></div>
<p class="key"><span class="adapter">adapter</span><span class="tso">TSO</span><span class="r1">Read1</span>{'<span class="false">site skera called an adapter</span>' if has_sreads else ''}
<span style="padding:0">lowercase: cDNA</span></p>"""]
    shown = {e["category"] for e in examples}
    missing = [TITLE[c] for c in CATEGORIES if c not in shown and (c != "false_cut" or has_sreads)]
    if not examples:
        out.append('<div class="prose"><p>No example reads: <code>--examples 0</code> was given, or no read checked qualifies.</p></div>')
    elif missing:
        out.append(f'<div class="prose"><p>No read checked qualifies as: {", ".join(missing)}.</p></div>')
    for e in examples:
        seq = list(e["seq"].lower())
        for _, a, b, kind in e["features"]:
            if kind in ("adapter", "false"):
                seq[a:b] = [x.upper() for x in seq[a:b]]
        focus = "" if e["focus"] is None else f' data-focus-line="{e["focus"] // WRAP}"'
        out.append(f'<section class="example" id="example-{e["category"]}-{e["zm"]}"><h3>{TITLE[e["category"]]}: ZMW {e["zm"]}, '
                   f'{e["read_len"]:,} bp</h3><p class="note">{_e(e["note"])}</p>'
                   f'<pre{focus}>{html_sequence("".join(seq), [(_e(n), a, b, k) for n, a, b, k in e["features"]], wrap=WRAP)}</pre></section>')
    return "\n".join(out)


def _versions() -> str:
    return ", ".join(f"{k} {importlib.metadata.version(k)}" for k in ("polars", "pysam", "edlib", "numpy", "plotly"))


def _section_inputs(s: dict, n_section: int) -> str:
    inp, ch = s["inputs"], s["chemistry"]
    rows = [["HiFi BAM", f"<code>{_e(inp['hifi'])}</code>"],
            ["skera S-read BAM", f"<code>{_e(inp['sreads'])}</code>" if inp["sreads"] else "not given"],
            ["Adapters", _e(ch["adapters"]) + f" ({_e(' '.join(ch['order']))})"],
            ["Primers", f"{_e(ch['primers'])}: TSO <code>{_e(ch['tso'])}</code>, Read1 <code>{_e(ch['r1'])}</code>"],
            ["Fraction of ZMWs", f"{s['fraction']} " + _key("fraction", s["fraction"])],
            ["HiFi reads in the file", f"{s['hifi_reads_in_file']:,} " + _key("hifi_reads_in_file", s["hifi_reads_in_file"])],
            ["HiFi reads checked", f"{s['hifi_reads']:,} " + _key("hifi_reads", s["hifi_reads"])]]
    if "molecules_written" in s:
        rows.append(["Molecules written to molecules.bam, from every read in the file", f"{s['molecules_written']:,}"])
    sk = f"skera {_e(s['skera_version'] or 'n/a')} (from the S-read BAM's header), " if inp["sreads"] else ""
    rows.append(["Software", f"kinnex-qc {_e(s['version'])}, {sk}Python {platform.python_version()}, {_versions()}"])
    whole = ('"HiFi reads in the file" (in this table and under the title) and "Molecules written to molecules.bam" count'
             if "molecules_written" in s else '"HiFi reads in the file" (in this table and under the title) counts')
    fraction = ("Every ZMW in the file was checked." if s["fraction"] >= 1 else
                f"A ZMW is checked when a salted hash of its number falls in the lowest {_pct(s['fraction'])} of the hash range, so the "
                f"sample spreads over the whole cell whatever the order of the file. Only {whole} the whole file; every other number "
                "above describes that sample.")
    limits = [
        "lima, isoseq refine and mapping are not run: the report does not say which S-reads or molecules survive the standard workflow.",
        "Fail reads (ZMWs whose consensus did not reach HiFi quality) are not checked.",
        "Molecule lengths and the order in which molecules of different lengths join (concatenation bias) are not checked.",
        "The cell label and UMI after Read1 are not decoded; the report does not say which cell a molecule belongs to.",
    ]
    if "skera" in s:
        limits.append("skera's own verdict file, non_passing.bam, is not read: passing is worked out from the S-read labels. On the three "
                      "BD144 libraries this agreed with skera's own verdict on every complete array (observed, skera 26.1.0).")
    else:
        limits.append("skera is not compared, since no S-read BAM was given (--sreads): no false cuts, mirror arms or slot budget.")
    return f"""<h2 id="inputs"><span class="sec">{n_section}</span>Inputs, versions and what was not checked</h2>
{_table(['Item', 'Value'], rows)}
<div class="prose"><p>{fraction}</p>
<h3>Not checked</h3><ul>{''.join(f'<li>{x}</li>' for x in limits)}</ul>
<p>The numbers come from <code>summary.json</code> in the same folder; the key is given beside a number or under its table.
<code>reads.tsv.gz</code> has the results read by read.</p></div>"""


def write(out_dir: Path, summary: dict, examples: list[dict]) -> Path:
    """Write out_dir/report.html from summary (summary.json's content, NaN as None) and the example reads
    (pick_examples' dicts, each with its read's sequence under "seq"); return its path."""
    has_skera = "skera" in summary
    body = [_section_checks(summary), _section_structure(summary), _section_artifacts(summary)]
    if has_skera:
        body.append(_section_skera(summary))
    body.append(_section_examples(summary, examples, 5 if has_skera else 4))
    body.append(_section_inputs(summary, 6 if has_skera else 5))
    js = (plotly.offline.get_plotlyjs()
          .replace("\ufffd", "\\ufffd").replace("\u2013", "\\u2013").replace("\u2014", "\\u2014"))   # these sit in JS strings only
    page = Template(files("kinnex_qc").joinpath("templates/reader_report.html").read_text(encoding="utf-8")).substitute(
        title=_e(f"kinnex-qc report: {Path(summary['inputs']['hifi']).name}"), viewer_css=VIEWER_CSS, plotly_js=js,
        header=_header(summary), toc=_toc(has_skera), body="\n".join(body))
    path = Path(out_dir) / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
