"""Supplementary Figs. 2 to 9 of the Note as plotly figures, built from the report's source tables and numbers.json (load),
for the render step. Fig. 3 draws the example reads of fig3_reads.json with the read-track code (read_tracks.py).
Figs. 1 and 4a are drawings, not data: kinnex_qc.schematics."""
import math
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from compute import ARG, LETTERS, ORDER3
from kinnex_qc.scan import Hit
from kinnex_qc.schematics import FATE
from read_tracks import read_track_figure

COLOR = {"ND": "#D55E00", "HM": "#0072B2", "kin": "#009E73", ARG: "#882255"}
INK, INK2, RULE, GRID = "#1B1F24", "#5B636E", "#C9D0D7", "#EDF0F3"
NEUTRAL = ["#E2E7EC", "#A7B1BB", "#5E6A76", "#2B333B"]       # light to dark, one cool-biased neutral ramp
NULL_P, NULL_MT = "rgba(91,99,110,0.22)", "rgba(31,95,139,0.20)"
FONT = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"


def load(tables: Path, numbers: dict) -> dict:
    """The figure inputs, read back from the source tables in tables/ and from numbers.json."""
    def read(name: str) -> pl.DataFrame:
        return pl.read_csv(tables / f"{name}.tsv", separator="\t", infer_schema_length=None)

    def by_library(name: str) -> dict[str, pl.DataFrame]:
        t = read(name)
        return {lib: t.filter(pl.col("library") == lib).drop("library") for lib in ORDER3}

    letters = read("fig9c_tso_tso_junction_letters")
    return {"pattern": read("fig2a_chain_patterns").to_dicts(), "artifacts": read("fig2b_artifact_rates").to_dicts(),
            "junction_letters": read("fig4b_mirror_junction_letters_ND"), "mirror_rule": read("fig4c_mirror_array_size_rule_ND"),
            "mirror_rule_overall": numbers["mirror_rule_overall"], "mirror_rule_reads": numbers["mirror_rule_reads"],
            "yield": read("fig5a_yield").to_dicts(), "decomposition": read("fig5c_ND_gap_decomposition").row(0, named=True),
            "budget": {r["library"]: r for r in read("fig6_molecule_budget").iter_rows(named=True)},
            "partner_curves": by_library("fig7_partner_curves"), "lag": by_library("fig8a_lag_profiles"),
            "selection": by_library("fig8b_length_selection"),
            "arg_full": read("S7_argentag_parent_reads").row(0, named=True), "arg_validation": read("S8_argentag_flag_validation").row(0, named=True),
            "tso_tso_letters": {lib: dict(letters.filter(pl.col("library") == lib).select("letter", "junctions").iter_rows())
                                for lib in ("ND", ARG)},
            "sread_tests": read("S8_sread_tests")}


def style(fig: go.Figure, height: int, width: int = 1000) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, width=width, font=dict(family=FONT, size=12, color=INK2),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", margin=dict(l=70, r=30, t=50, b=60),
                      legend=dict(font=dict(size=11, color=INK2), bgcolor="rgba(255,255,255,0)", traceorder="normal"),
                      hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor="#FFFFFF", bordercolor=RULE))
    fig.update_xaxes(showline=True, linecolor=INK2, linewidth=1, ticks="outside", tickcolor=INK2, gridcolor=GRID, zeroline=False,
                     title_font=dict(size=12, color=INK2))
    fig.update_yaxes(showline=True, linecolor=INK2, linewidth=1, ticks="outside", tickcolor=INK2, gridcolor=GRID, zeroline=False,
                     title_font=dict(size=12, color=INK2))
    titles = [a for a in fig.layout.annotations if a.xref == "paper" and str(a.text).startswith("<b>")]
    for i, ann in enumerate(titles):                         # subplot titles are the panel letters: left-align them on their panel
        ax = fig.layout["xaxis" if i == 0 else f"xaxis{i + 1}"]
        ann.update(font=dict(size=14, color=INK, family=FONT), xanchor="left", x=ax.domain[0], align="left")
    return fig


def panel_titles(*t):
    return [f"<b>{x}</b>" for x in t]


def fig2(d):
    f = make_subplots(rows=1, cols=2, column_widths=[0.52, 0.48], horizontal_spacing=0.1,
                      subplot_titles=panel_titles("a  Adapter chain of each HiFi read", "b  Artifact molecules per 1,000 molecules"))
    pats = [("complete", "Complete, A to Q", NEUTRAL[0]), ("truncated", "Truncated", NEUTRAL[1]), ("other", "Other", NEUTRAL[2]),
            ("mirror", "Mirror junction", NEUTRAL[3])]
    p = {(r["library"], r["pattern"]): r for r in d["pattern"]}
    for key, lab, col in pats:
        xs = [100 * p.get((lib, key), {"share": 0})["share"] for lib in ORDER3]
        f.add_trace(go.Bar(y=ORDER3, x=xs, orientation="h", name=lab, marker=dict(color=col, line=dict(color="#FFFFFF", width=1.5)),
                           hovertemplate=f"{lab}: %{{x:.2f}}% of HiFi reads<extra>%{{y}}</extra>", legendgroup="pat"), 1, 1)
    for lib in ORDER3:
        m = 100 * p[(lib, "mirror")]["share"]
        f.add_annotation(x=101, y=lib, text=f"mirror {m:.2f}%" if m < 1 else f"mirror {m:.1f}%", showarrow=False, xanchor="left",
                         font=dict(size=11, color=INK), row=1, col=1)
    f.update_xaxes(range=[0, 122], tickvals=[0, 20, 40, 60, 80, 100], title_text="HiFi reads (%)", row=1, col=1)
    f.update_yaxes(autorange="reversed", row=1, col=1)
    classes = ["TSO-TSO", "R1-R1", "TSO only", "R1 only", "Fused (no adapter between)"]
    short = {"Fused (no adapter between)": "Fused"}
    a = {(r["library"], r["class"]): r["per_1000_molecules"] for r in d["artifacts"]}
    for i, lib in enumerate(ORDER3):
        f.add_trace(go.Scatter(x=[max(a[(lib, c)], 1e-3) for c in classes], y=[short.get(c, c) for c in classes], mode="markers", name=lib,
                               marker=dict(color=COLOR[lib], size=10, line=dict(color="#FFFFFF", width=1.5)),
                               customdata=[a[(lib, c)] for c in classes],
                               hovertemplate="%{y}: %{customdata:.3f} per 1,000<extra>" + lib + "</extra>", legendgroup="lib"), 1, 2)
    f.update_xaxes(type="log", title_text="per 1,000 molecules (log scale)", tickvals=[0.001, 0.01, 0.1, 1, 10], row=1, col=2,
                   ticktext=["0.001", "0.01", "0.1", "1", "10"])
    f.update_yaxes(autorange="reversed", row=1, col=2)
    f.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.22, x=0))
    return style(f, 380)


def track_hits(read: dict) -> dict:
    """One read of fig3_reads.json as read_track_figure takes it: hits as kinnex_qc.scan.Hit, S-reads as (qs, qe)."""
    return {**read, "features": [Hit(**h) for h in read["features"]], "skera": [tuple(x) for x in read["skera"]]}


def fig3(reads: list[dict]) -> go.Figure:
    """The example reads of fig3_reads.json, one track each, in the order given."""
    fig = read_track_figure([track_hits(r) for r in reads], "")
    fig.update_layout(title=dict(text="", subtitle=dict(text="")), font=dict(family=FONT), width=1040, height=760,
                      margin=dict(l=80, r=215, t=30, b=60))
    return fig


def fig4(d):
    f = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.12,
                      subplot_titles=panel_titles("b  Adapter at each mirror junction (ND)", "c  Molecules per mirror array (ND)"))
    jl = d["junction_letters"]
    for mol, col in (("TSO-TSO", COLOR["ND"]), ("R1-R1", NEUTRAL[2]), ("other", NEUTRAL[1])):
        sub = {r["letter"]: r["junctions"] for r in jl.filter(pl.col("molecule") == mol).iter_rows(named=True)}
        f.add_trace(go.Bar(x=list(LETTERS), y=[sub.get(L, 0) / 1000 for L in LETTERS], name=f"{mol} at the junction",
                           marker=dict(color=col, line=dict(color="#FFFFFF", width=1)),
                           hovertemplate="adapter %{x}: %{y:.2f} thousand junctions<extra>" + mol + "</extra>"), 1, 1)
    f.update_layout(barmode="stack")
    f.update_xaxes(title_text="adapter on both sides of the junction", row=1, col=1)
    f.update_yaxes(title_text="mirror junctions (thousands)", row=1, col=1)
    mr = d["mirror_rule"].filter(pl.col("letter").is_in(list(LETTERS[1:])))
    f.add_trace(go.Scatter(x=mr["letter"].to_list(), y=mr["predicted"].to_list(), mode="lines+markers", name="predicted, 2 x (slots from X to Q) + 1",
                           line=dict(color=INK2, width=2, dash="dot"), marker=dict(size=8, symbol="circle-open", color=INK2)), 1, 2)
    f.add_trace(go.Scatter(x=mr["letter"].to_list(), y=mr["median_molecules"].to_list(), mode="markers", name="observed median",
                           marker=dict(size=10, color=COLOR["ND"], line=dict(color="#FFFFFF", width=1.5)),
                           customdata=list(zip(mr["reads"].to_list(), (100 * mr["share_matching"]).to_list())),
                           hovertemplate="junction at %{x}: median %{y} molecules, %{customdata[0]:,} reads, %{customdata[1]:.1f}% match<extra></extra>"), 1, 2)
    f.add_annotation(x="E", y=15.5, text=f"{100 * d['mirror_rule_overall']:.1f}% of {d['mirror_rule_reads']:,} single-junction reads match",
                     showarrow=False, font=dict(size=11, color=INK), row=1, col=2)
    f.update_xaxes(title_text="junction adapter X", row=1, col=2)
    f.update_yaxes(title_text="molecules in the array", range=[0, 17], dtick=2, row=1, col=2)
    f.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
    return style(f, 400)


def fig5(d):
    f = make_subplots(rows=1, cols=3, column_widths=[0.3, 0.38, 0.32], horizontal_spacing=0.1,
                      subplot_titles=panel_titles("a  Molecules per HiFi read", "b  skera on ND reads, by chain", "c  ND against kin"))
    y = {r["library"]: r for r in d["yield"]}
    for metric, lab, sym in (("proper_per_hifi_read", "our segmentation", "circle"), ("mapped_fltnc_per_hifi_zmw", "skera, lima, refine, mapped", "diamond")):
        f.add_trace(go.Scatter(x=ORDER3, y=[y[lib][metric] for lib in ORDER3], mode="markers", name=lab, showlegend=False,
                               marker=dict(size=12, symbol=sym, color=[COLOR[lib] for lib in ORDER3], line=dict(color=INK, width=1)),
                               hovertemplate="%{x}: %{y:.3f} molecules per HiFi read<extra>" + lab + "</extra>"), 1, 1)
        f.add_annotation(x="HM", y=(y["HM"][metric] + y["kin"][metric]) / 2 + (0.09 if sym == "circle" else -0.09), text=lab, showarrow=False,
                         font=dict(size=11, color=INK2), row=1, col=1)
    lo = min(min(y[lib]["proper_per_hifi_read"], y[lib]["mapped_fltnc_per_hifi_zmw"]) for lib in ORDER3)
    f.update_yaxes(title_text="molecules per HiFi read", range=[math.floor(lo * 5) / 5 - 0.1, 8.2], row=1, col=1)
    p = {(r["library"], r["pattern"]): r for r in d["pattern"]}
    pats = ["complete", "truncated", "mirror"]
    f.add_trace(go.Bar(x=pats, y=[p[("ND", k)]["proper_per_read"] for k in pats], name="molecules found by our segmentation",
                       offsetgroup="ours", marker=dict(color="#FFFFFF", line=dict(color=NEUTRAL[2], width=1.5)), hovertemplate="%{x}: %{y:.2f} molecules per read<extra></extra>"), 1, 2)
    f.add_trace(go.Bar(x=pats, y=[p[("ND", k)]["skera_sreads_per_read"] for k in pats], name="S-reads skera writes",
                       offsetgroup="skera", marker=dict(color=NEUTRAL[3]), hovertemplate="%{x}: %{y:.2f} S-reads per read<extra></extra>"), 1, 2)
    f.update_xaxes(tickvals=pats, ticktext=[f"{k}<br>passes {100 * p[('ND', k)]['skera_pass_share']:.1f}%" for k in pats], tickangle=0,
                   tickfont=dict(size=11), row=1, col=2)
    f.update_yaxes(title_text="per HiFi read", range=[0, 8.6], row=1, col=2)
    dc = d["decomposition"]
    parts = [("Fewer molecules in mirror arrays", dc["mirror_capacity"], dict(color=FATE["artifact"], line=dict(color="#FFFFFF", width=1.5))),
             ("Mirror-array molecules skera never writes", dc["skera_extra"],
              dict(color="#FFFFFF", line=dict(color=FATE["unwritten"], width=1.5), pattern=dict(shape="/", fgcolor=FATE["unwritten"], size=7, solidity=0.45)))]
    base = 0.0
    for lab, v, marker in parts:                  # stacked by hand (base=): barmode is figure-wide and panel b needs "group"
        f.add_trace(go.Bar(x=["mapped fltnc gap"], y=[v], base=[base], name=lab, offsetgroup="gap", marker=marker,
                           hovertemplate=lab + ": %{y:.3f} per HiFi read<extra></extra>"), 1, 3)
        base += v
    f.add_trace(go.Bar(x=["our segmentation gap"], y=[dc["ours_gap"]], name="gap with our segmentation", showlegend=False, offsetgroup="gap",
                       marker=dict(color=FATE["artifact"], line=dict(color="#FFFFFF", width=1.5)),
                       hovertemplate="%{y:.3f} per HiFi read<extra></extra>"), 1, 3)
    f.add_annotation(x="mapped fltnc gap", y=dc["mapped_gap"] + 0.04, text=f"observed {dc['mapped_gap']:.2f}", showarrow=False,
                     font=dict(size=11, color=INK), row=1, col=3)
    f.add_annotation(x="our segmentation gap", y=dc["ours_gap"] + 0.04, text=f"observed {dc['ours_gap']:.2f}", showarrow=False,
                     font=dict(size=11, color=INK), row=1, col=3)
    f.update_yaxes(title_text="molecules per HiFi read, kin minus ND", range=[0, 1.2 * max(dc["mapped_gap"], dc["ours_gap"])], row=1, col=3)
    f.update_layout(barmode="group", bargroupgap=0.08, legend=dict(orientation="h", y=-0.3, x=0))
    return style(f, 430)


def fig6(d):
    """Per HiFi read, the slots whose molecule the standard workflow does not keep: greys are lost in either workflow, hatched amber
    only in the standard workflow (a full-read segmentation keeps them)."""
    names = {"ND": "ND, no depletion", "HM": "HM, home-made depletion", "kin": "kin, Kinnex depletion"}
    ys = [names[lib] for lib in ORDER3]
    b = {lib: d["budget"][lib] for lib in ORDER3}
    art = {lib: b[lib]["artifact_tso_tso"] + b[lib]["artifact_r1_r1"] + b[lib]["artifact_one_primer_or_none"] + b[lib]["artifact_fused"] for lib in ORDER3}
    down = {lib: b[lib]["lost_lima"] + b[lib]["lost_refine"] + b[lib]["lost_mapping"] for lib in ORDER3}
    sep = dict(color="#FFFFFF", width=1)
    segs = [("Slots the array does not fill", [b[lib]["unfilled_slots"] for lib in ORDER3], dict(color=FATE["unfilled"], line=sep)),
            ("Artifact molecules (TSO-TSO, R1-R1, one primer, fused)", [art[lib] for lib in ORDER3], dict(color=FATE["artifact"], line=sep)),
            ("Usable molecules lost at lima, refine or mapping", [down[lib] for lib in ORDER3], dict(color=FATE["downstream"], line=sep)),
            ("Usable molecules skera never writes cleanly: lost in the standard workflow, kept by a full-read segmentation",
             [b[lib]["not_written_cleanly"] for lib in ORDER3],
             dict(color="#FFFFFF", line=sep, pattern=dict(shape="/", fgcolor=FATE["unwritten"], bgcolor="#FFFFFF", size=8, solidity=0.5)))]
    f = go.Figure()
    for lab, vals, marker in segs:
        f.add_trace(go.Bar(y=ys, x=vals, orientation="h", name=lab, marker=marker, text=[f"{v:.2f}" if v >= 0.1 else "" for v in vals],
                           textposition="inside", insidetextanchor="middle", textfont=dict(size=12, color=INK),
                           hovertemplate="%{y}: %{x:.3f} per HiFi read<extra>" + lab + "</extra>"))
    top = 0.0
    for lib, y in zip(ORDER3, ys):
        total = b[lib]["unfilled_slots"] + art[lib] + down[lib] + b[lib]["not_written_cleanly"]
        top = max(top, total)
        f.add_annotation(x=total, y=y, xanchor="left", showarrow=False, align="left", font=dict(size=12, color=INK),
                         text=f"  standard workflow keeps {b[lib]['kept_standard']:.2f} of 8<br>  full-read segmentation keeps "
                              f"{b[lib]['kept_standard'] + b[lib]['not_written_cleanly']:.2f} of 8")
    f.update_yaxes(categoryorder="array", categoryarray=ys[::-1], tickfont=dict(size=12))
    f.update_xaxes(title_text="molecules per HiFi read, out of 8 slots, that the standard workflow does not keep", range=[0, top + 0.62], dtick=0.2)
    f.update_layout(barmode="stack", bargap=0.42, legend=dict(orientation="h", y=-0.32, x=0))
    return style(f, 430).update_layout(margin=dict(l=190))


def fig7(d):
    f = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, shared_yaxes=False,
                      subplot_titles=panel_titles("a  ND", "b  HM", "c  kin"))
    for i, lib in enumerate(ORDER3, start=1):
        t = d["partner_curves"][lib].filter(pl.col("n_index") >= 2000)
        x = ((t["bin_start"] + t["bin_end"]) / 2000).to_list()
        for lo, hi, col, name in (("P_mean_lo", "P_mean_hi", NULL_P, "null P: positions independent"),
                                  ("MT_mean_lo", "MT_mean_hi", NULL_MT, "M_T: independent positions plus selection on total length")):
            f.add_trace(go.Scatter(x=x + x[::-1], y=t[hi].to_list() + t[lo].to_list()[::-1], mode="lines", fill="toself", fillcolor=col,
                                   line=dict(width=0), name=name, showlegend=i == 1, legendgroup=name, hoverinfo="skip"), 1, i)
        f.add_trace(go.Scatter(x=x, y=t["partner_mean"].to_list(), mode="lines+markers", name="observed (library colour)", showlegend=i == 1, legendgroup="obs",
                               line=dict(color=COLOR[lib], width=2), marker=dict(size=6, color=COLOR[lib]),
                               customdata=t["n_index"].to_list(),
                               hovertemplate="index %{x:.2f} kb: partners %{y:.0f} bp (%{customdata:,.0f} index molecules)<extra>" + lib + "</extra>"), 1, i)
        f.update_xaxes(title_text="index molecule length (kb)", row=1, col=i)
    f.update_yaxes(title_text="mean partner length (bp)", row=1, col=1)
    f.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
    return style(f, 380)


def fig8(d):
    f = make_subplots(rows=2, cols=3, vertical_spacing=0.22, horizontal_spacing=0.07,
                      subplot_titles=panel_titles("a  ND", "b  HM", "c  kin", "d  ND", "e  HM", "f  kin"))
    for i, lib in enumerate(ORDER3, start=1):
        t = d["lag"][lib]
        lag = t["lag"].to_list()
        f.add_trace(go.Scatter(x=lag + lag[::-1], y=t["MT_hi"].to_list() + t["MT_lo"].to_list()[::-1], mode="lines", fill="toself", fillcolor=NULL_MT,
                               line=dict(width=0), name="M_T band", showlegend=i == 1, legendgroup="mt", hoverinfo="skip"), 1, i)
        f.add_trace(go.Scatter(x=lag, y=t["observed"].to_list(), mode="markers", name="observed, bootstrap 95% CI (library colour)", showlegend=i == 1,
                               legendgroup="obs", marker=dict(size=8, color=COLOR[lib]),
                               error_y=dict(type="data", symmetric=False, array=(t["boot_hi"] - t["observed"]).to_list(),
                                            arrayminus=(t["observed"] - t["boot_lo"]).to_list(), color=COLOR[lib], thickness=1.2, width=3),
                               hovertemplate="distance %{x}: %{y:.4f}<extra>" + lib + "</extra>"), 1, i)
        f.update_xaxes(title_text="distance between positions in the array", dtick=1, row=1, col=i)
        s = d["selection"][lib].filter(pl.col("complete_arrays") > 500)
        x = ((s["T_bin_start"] + s["T_bin_end"]) / 2000).to_list()
        for col_name, lab, col, dash in (("s(T) from M_T", "s(T): selection on total length", COLOR[lib], "solid"),
                                         ("w(T) empirical", "w(T): HiFi yield, empirical", INK2, "dash"),
                                         ("w(T) model", "w(T): HiFi yield, model", INK2, "dot")):
            f.add_trace(go.Scatter(x=x, y=s[col_name].to_list(), mode="lines", name=lab, showlegend=i == 1, legendgroup=lab,
                                   line=dict(color=col if col_name.startswith("s") else INK2, width=2.2 if col_name.startswith("s") else 1.6, dash=dash),
                                   hovertemplate="%{x:.1f} kb: %{y:.3f}<extra>" + lab + "</extra>"), 2, i)
        f.update_xaxes(title_text="array length (kb)", row=2, col=i)
        f.update_yaxes(range=[0, 1.05], row=2, col=i)
    f.update_yaxes(title_text="rank correlation of lengths", row=1, col=1)
    f.update_yaxes(title_text="relative probability (max 1)", row=2, col=1)
    f.update_layout(legend=dict(orientation="h", y=-0.14, x=0))
    return style(f, 680)


def fig9(d):
    order4 = ORDER3 + [ARG]
    af, av = d["arg_full"], d["arg_validation"]
    pat = {(r["library"], r["pattern"]): r for r in d["pattern"]}
    art = {(r["library"], r["class"]): r["per_1000_molecules"] for r in d["artifacts"]}
    mirror = {lib: pat[(lib, "mirror")]["share"] * 100 for lib in ORDER3} | {ARG: af["mirror_share"] * 100}
    tt = {lib: art[(lib, "TSO-TSO")] for lib in ORDER3} | {ARG: af["tso_tso_per_1000"]}
    f = make_subplots(rows=2, cols=2, horizontal_spacing=0.13, vertical_spacing=0.24,
                      subplot_titles=panel_titles("a  Mirror reads", "b  TSO-TSO molecules", "c  Adapter at TSO-TSO mirror junctions",
                                                  "d  S-read flag against the parent reads"))
    for col, vals, unit in ((1, mirror, "% of HiFi reads"), (2, tt, "per 1,000 molecules")):
        f.add_trace(go.Scatter(x=order4, y=[vals[lib] for lib in order4], mode="markers+text",
                               text=[f"{vals[lib]:.1f}" if vals[lib] >= 1 else f"{vals[lib]:.2g}" for lib in order4],
                               textposition="middle right", textfont=dict(size=10, color=INK2), showlegend=False,
                               marker=dict(size=13, color=[COLOR[lib] for lib in order4], line=dict(color="#FFFFFF", width=1.5)),
                               hovertemplate="%{x}: %{y:.4g} " + unit + "<extra></extra>"), 1, col)
        f.update_yaxes(type="log", title_text=f"{unit} (log scale)", row=1, col=col)
        f.update_xaxes(type="category", categoryorder="array", categoryarray=order4, range=[-0.5, 3.9], row=1, col=col)
    for lib in ("ND", ARG):
        src = d["tso_tso_letters"][lib]
        tot = sum(src.values())
        f.add_trace(go.Scatter(x=list("BCDEFGHQ"), y=[100 * src.get(L, 0) / tot for L in "BCDEFGHQ"], mode="lines+markers", name=lib,
                               legendgroup="lib", line=dict(color=COLOR[lib], width=2), marker=dict(size=7, color=COLOR[lib]),
                               hovertemplate="adapter %{x}: %{y:.1f}% of TSO-TSO junctions<extra>" + lib + "</extra>"), 2, 1)
    f.update_xaxes(title_text="adapter on both sides of the junction", row=2, col=1)
    f.update_yaxes(title_text="% of TSO-TSO junctions", row=2, col=1)
    t = {r["library"]: r for r in d["sread_tests"].iter_rows(named=True)}
    bg = [t[lib]["flag_non_mirror"] * 100 for lib in ORDER3] + [av["flag_non_mirror_share"] * 100]
    f.add_shape(type="rect", xref="x4 domain", x0=0, x1=1, y0=min(bg), y1=max(bg) * 1.15, fillcolor=NULL_P, line=dict(width=0), row=2, col=2)
    f.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="flagged ZMWs (library colour)", legendgroup="flag",
                           marker=dict(size=13, color=INK2, line=dict(color=INK, width=1))), 2, 2)
    f.add_trace(go.Scatter(x=order4, y=[t[lib]["flag"] * 100 for lib in order4], mode="markers", showlegend=False,
                           marker=dict(size=13, color=[COLOR[lib] for lib in order4], line=dict(color=INK, width=1)),
                           hovertemplate="%{x}: %{y:.3f}% of HiFi reads flagged<extra></extra>"), 2, 2)
    f.add_trace(go.Scatter(x=order4, y=[mirror[lib] for lib in order4], mode="markers", name="mirror reads, parent reads", legendgroup="flag",
                           marker=dict(size=13, symbol="circle-open", color=INK, line=dict(width=1.5)),
                           hovertemplate="%{x}: %{y:.3f}% mirror reads<extra></extra>"), 2, 2)
    f.add_annotation(x=1, xref="x4 domain", y=math.log10(max(bg) * 1.15), text="non-mirror reads flagged ", showarrow=False, xanchor="right",
                     yanchor="bottom", font=dict(size=10, color=INK2), row=2, col=2)
    f.update_yaxes(type="log", range=[-2.3, 1.4], tickvals=[0.01, 0.1, 1, 10], ticktext=["0.01", "0.1", "1", "10"],
                   title_text="% of HiFi reads (log scale)", row=2, col=2)
    f.update_xaxes(type="category", categoryorder="array", categoryarray=order4, row=2, col=2)
    f.update_layout(legend=dict(orientation="h", y=-0.14, x=0))
    return style(f, 760)
