#!/usr/bin/env python3
"""The render step of the Supplementary Note: from what compute.py writes into <run>/report/ (tables/, numbers.json and
fig3_reads.json) builds, in the same folder,
  supplementary_note.html        the Note: plotly figures, the plotly bundle inlined once,
  supplementary_note.print.html  the same text with the figures as SVG files and print.css, for printing,
  supplementary_note.pdf         the print page as PDF (headless Chrome),
  figures/                       each figure as SVG and PDF (plotly figures through kaleido, the two drawings through Inkscape).
The prose is template.html. A placeholder {key|format} prints the numbers.json value of key with one of FORMATS (v10's
helpers), a range {lo_key..hi_key|format} "lo to hi" or one value when both print the same (v10's span());
<!--table S1--> and the like mark where the Supplementary Tables go, which render builds from tables/ and
numbers.json; <!--figure fig2--> a plotly figure of figures.py; <!--schematic fig1--> a drawing of kinnex_qc.schematics.
Usage: study/report/render.py --report-dir <run>/report
"""
import argparse
import base64
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import plotly
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

import figures
from common import ARG, KEY, ORDER3
from kinnex_qc.schematics import false_cut_svg, schematic_svg

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template.html"
PRINT_CSS = HERE / "print.css"
FONT_DIR = HERE.parent.parent / "resources" / "fonts"
FIGURES = ("fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9")
SCHEMATICS = {"fig1": "Supplementary_fig1", "fig4a": "Supplementary_fig4a_schematic"}      # slot -> file name in figures/
# The fonts v10 loaded from Google Fonts, inlined in both pages: family, weight, style, file in resources/fonts (README.md
# there names the releases). Source Serif 4 in the static cuts of the optical sizes v10 set it at: Small Text, Subhead.
FONTS = [("IBM Plex Sans", 400, "normal", "IBMPlexSans-Regular.woff2"), ("IBM Plex Sans", 500, "normal", "IBMPlexSans-Medium.woff2"),
         ("IBM Plex Sans", 600, "normal", "IBMPlexSans-SemiBold.woff2"), ("IBM Plex Mono", 400, "normal", "IBMPlexMono-Regular.woff2"),
         ("IBM Plex Mono", 500, "normal", "IBMPlexMono-Medium.woff2"), ("Source Serif 4", 400, "normal", "SourceSerif4SmText-Regular.ttf.woff2"),
         ("Source Serif 4", 600, "normal", "SourceSerif4SmText-Semibold.ttf.woff2"), ("Source Serif 4", 400, "italic", "SourceSerif4SmText-It.ttf.woff2"),
         ("Source Serif 4 Subhead", 400, "normal", "SourceSerif4Subhead-Regular.ttf.woff2"),
         ("Source Serif 4 Subhead", 600, "normal", "SourceSerif4Subhead-Semibold.ttf.woff2")]


def fmt(v, digits=2):
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.{digits}f}"


def sig2(v):
    """Round to two significant figures, with thousands separators (207 -> 210, 2978 -> 3,000)."""
    return f"{round(v, 1 - int(math.floor(math.log10(abs(v))))):,.0f}"


def small_pct(v):
    """A share below 1% to two significant figures (0.095%), as the figures label it."""
    return f"{100 * v:.2g}%"


def pct(v, digits=1):
    return f"{100 * v:.{digits}f}%"


def commits(v: list[str]) -> str:
    return ", ".join(v[:-1]) + " and " + v[-1] if len(v) > 1 else "".join(v)


# The formats a placeholder may name: v10's helpers and the format specs its text used.
FORMATS = {
    "fmt": fmt, "int": lambda v: f"{v:,}", "comma0": lambda v: f"{v:,.0f}",
    "pct0": lambda v: pct(v, 0), "pct1": pct, "pct2": lambda v: pct(v, 2), "small_pct": small_pct, "sig2": sig2,
    "pp1": lambda v: f"{100 * v:.1f}",                                  # a share as percentage points
    "f0": lambda v: f"{v:.0f}", "f1": lambda v: f"{v:.1f}", "f2": lambda v: f"{v:.2f}", "f3": lambda v: f"{v:.3f}",
    "sf1": lambda v: f"{v:+.1f}", "sf3": lambda v: f"{v:+.3f}", "sf4": lambda v: f"{v:+.4f}", "g2": lambda v: f"{v:.2g}",
    "kb1": lambda v: f"{v / 1000:.1f}", "mio2": lambda v: f"{v / 1e6:.2f}",
    "str": str, "list": lambda v: ", ".join(str(x) for x in v), "commits": commits,
}
PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)(?:\.\.([a-z0-9_]+))?\|([a-z0-9_]+)\}")
BARE = re.compile(r"\{([a-z0-9_]+)\}")


def span(lo: str, hi: str) -> str:
    """v10's span(): a range as "lo to hi", or one value when both ends print the same."""
    return lo if lo == hi else f"{lo} to {hi}"


def placeholders(text: str) -> list[tuple[str, str]]:
    """(key, format) of every placeholder in text, in order; a range gives both its keys."""
    return [(key, f) for lo, hi, f in PLACEHOLDER.findall(text) for key in (lo, hi) if key]


def fill(text: str, numbers: dict) -> str:
    """text with every {key|format} replaced by FORMATS[format](numbers[key]), and every range {lo..hi|format} by span() of
    its two ends in that format. A key that numbers.json lacks or a format not in FORMATS is a KeyError; a {key} without a
    format a ValueError."""
    bare = BARE.search(text)
    if bare:
        raise ValueError(f"placeholder {bare.group(0)} names no format; write {{{bare.group(1)}|<format>}}, format one of {', '.join(FORMATS)}")

    def one(m: re.Match) -> str:
        lo, hi, f = m.groups()
        for key in (lo, hi):
            if key and key not in numbers:
                raise KeyError(f"{key}: not a key of numbers.json")
        if f not in FORMATS:
            raise KeyError(f"{f} (in {m.group(0)}): not a format, use one of {', '.join(FORMATS)}")
        return span(FORMATS[f](numbers[lo]), FORMATS[f](numbers[hi])) if hi else FORMATS[f](numbers[lo])

    return PLACEHOLDER.sub(one, text)


def read(tables: Path, name: str) -> pl.DataFrame:
    return pl.read_csv(tables / f"{name}.tsv", separator="\t", infer_schema_length=None)


def check_claims(tables: Path, numbers: dict) -> None:
    """The template's words that hold only for these data, checked against the source tables and numbers.json. When one
    does not hold, stop with one line naming it: the Note's text then needs a change in template.html. "Alike" is read as
    within 10% of each other and "slight" as under 1% (thresholds of this check, not of v10)."""
    fa = {r["library"]: r for r in read(tables, "fig1_skera_false_cut_anatomy").iter_rows(named=True)}
    lima = max(abs(fa[lib][col] - fa["kin"][col]) for lib in ("ND", "HM") for col in ("fate_fused_lima", "fate_frag_lima"))
    partner = max(abs(r["diff_vs_P_bp"]) / r["partner_mean"] for r in read(tables, "S6_concatenation_bias").iter_rows(named=True))
    bench = read(tables, "S4_chain_classification_benchmark_summary").filter(pl.col("error_rate") <= 0.01)    # HiFi rates, as bench_hifi_class_min
    false_mirrors = bench["non_mirror_called_mirror"].sum()
    arg_at_a = read(tables, "S7_argentag_parent_reads")["junctions_at_A_tso_tso"].sum()
    let = read(tables, "fig4b_mirror_junction_letters_ND")
    tt_at_a, rr_at_q = (let.filter((pl.col("letter") == x) & (pl.col("molecule") == m))["junctions"].sum() for x, m in (("A", "TSO-TSO"), ("Q", "R1-R1")))
    fc_lo, fc_hi, passed = numbers["false_cut_reads_share_min"], numbers["false_cut_reads_share_max"], numbers["nd_tso_tso_only_skera_pass_share"]
    claims = [
        ("Supplementary Fig. 1's caption", "ND and HM differ by less than 0.1 percentage point", lima < 0.001,
         f"their lima pass rates differ from kin's by up to {100 * lima:.3f} percentage point"),
        ("the key findings and section 5", "under 1% of the mean partner", partner < 0.01,
         f"the partners of long molecules differ from chance by {pct(partner, 2)} of the mean partner in one library"),
        ("section 1", "never calls a read a mirror array when it is not one", false_mirrors == 0,
         f"non-mirror reads called mirror at HiFi error rates in the benchmark: {false_mirrors}"),
        ("section 6", "none of them at adapter A", arg_at_a == 0, f"ArgenTag TSO-TSO junctions at A: {arg_at_a}"),
        ("section 3", "junctions occur at adapters B to Q but not at A", tt_at_a == 0, f"ND TSO-TSO junctions at A: {tt_at_a}"),
        ("section 3", "with junctions at A to H but never at Q", rr_at_q == 0, f"ND R1-R1 junctions at Q: {rr_at_q}"),
        ("section 1", "in all three libraries alike", fc_hi < 1.1 * fc_lo,
         f"the libraries' shares of reads with a false cut, {pct(fc_lo)} to {pct(fc_hi)}, are 10% or more apart"),
        ("section 6", "a slight underestimate", passed < 0.01, f"skera passed {pct(passed)} of ND reads whose only junction molecule is TSO-TSO"),
    ]
    for where, claim, holds, but in claims:
        if not holds:
            sys.exit(f'render.py: "{claim}" ({where}) does not hold: {but}; change the text in template.html')


def table_html(header, rows, cls="") -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


# Supplementary Table 4: row label, the synthetic read class of compute.BENCH_CLASSES (numbers.json bench_<class>_<rate>),
# and the call it should get (None: anything but mirror).
BENCH_ROWS = [("Complete arrays", "complete", "complete"), ("TSO-TSO mirrors, junction at B to Q", "tso_tso_mirror", "mirror"),
              ("R1-R1 mirrors, junction at A to H", "r1_r1_mirror", "mirror"), ("Partial, A end missing", "truncated_no_a", "truncated"),
              ("Partial, Q end missing", "truncated_no_q", "truncated"), ("One adapter destroyed (missed)", "adapter_destroyed", "skipped_slot"),
              ("D-like motifs inside every molecule", "decoy_motif", "complete"),
              ("Partial arrays joined in opposite orientation", "opposite_join", None)]
MIRROR_REGIONS = {"after": "Arm after the junction (X to Q, forward adapters)", "before": "Arm before the junction (Q' to X', reverse-complement adapters)",
                  "all_mirror_arms": "All proper molecules of these mirror reads", "complete_array": "Complete arrays, for comparison"}
MIRROR_CLASSES = ("clean", "fused", "partial", "not_written")


def supplementary_tables(tables: Path, n: dict) -> dict[str, str]:
    """Supplementary Tables 1 and 3 to 10 as v10 printed them (Table 2 is prose, in the template). Cells that are values of
    a source table come from tables/ (Tables 1, 3, 5, 6, 7 and 9), cells compute derived from numbers.json (4, 8 and 10)."""
    lib = list(read(tables, "S1_libraries").iter_rows(named=True))
    st = {r["library"]: r for r in read(tables, "S8_sread_tests").iter_rows(named=True)}
    rows = [[f"BD144 {x['short']}", x["cell"], x["depletion"], fmt(x["hifi_cell"]), fmt(x["hifi_analysed"]), fmt(x["skera_sreads"]),
             f"{x['sreads_per_read_sample']:.2f} / {x['sreads_per_read_cell']:.2f}", f"{pct(x['passing_sample'])} / {pct(x['full_array_cell'])}"]
            for x in lib]
    rows.append(["ArgenTag MDA-MB-453 (public)", "Revio, 8-fold", "None (described as not required)", fmt(st[ARG]["hifi_zmws"]),
                 "S-reads only", fmt(st[ARG]["sreads"]), "", ""])
    t = {"S1": table_html(["Library", "SMRT Cell", "TSO-TSO depletion", "HiFi reads in cell", "HiFi reads analysed", "skera S-reads analysed",
                           "S-reads per HiFi read: sample / whole cell", "skera passing reads, sample / full arrays, whole cell"], rows, "num")}

    rows = [[f"BD144 {g['library']}", f"{g['synthetic_recall_min']:.4f} / {g['synthetic_precision_min']:.4f}", f"{g['adapter_index_agreement']:.4f}",
             f"{g['lima_class_pair_agreement']:.4f}", f"{g['fltnc_in_lima_pass']:.4f}", f"{g['skera_agreement']:.3f}"]
            for g in read(tables, "S3_validation_gates").iter_rows(named=True)]
    t["S3"] = table_html(["Library", "Synthetic arrays: recall / precision", "Adapter letters agree with skera", "lima pair agrees with our class",
                          "fltnc in lima pass", "skera agreement"], rows, "num")

    bench = sorted(read(tables, "S4_chain_classification_benchmark_summary").iter_rows(named=True), key=lambda r: r["error_rate"])
    rates = [r["error_rate"] for r in bench]
    rows = [[label, "not mirror" if want is None else want.replace("_", " "),
             *[pct(n[f"bench_{name}_{r:g}".replace(".", "_")]) for r in rates]] for label, name, want in BENCH_ROWS]
    rows.append(["Mirror calls: precision", "", *[f"{r['mirror_precision']:.3f}" for r in bench]])
    t["S4"] = table_html(["Synthetic read class", "Expected call", *[f"error {r:g}" for r in rates]], rows, "num")

    rows = [[r["junction_molecules"], fmt(r["reads"]), f"{r['median_molecules']:.0f}", pct(r["skera_pass_share"])]
            for r in read(tables, "S5_mirror_junction_composition_ND").iter_rows(named=True)]
    t["S5"] = table_html(["Molecule at the mirror junction (ND)", "Reads", "Median molecules per read", "Passed by skera"], rows, "num")

    rows = [[r["library"], fmt(r["index_ge_4kb"]), f"{r['partner_mean']:.0f}", f"{r['null_P_mean']:.0f} [{r['null_P_lo']:.0f}, {r['null_P_hi']:.0f}]",
             f"{r['M_T_mean']:.0f} [{r['M_T_lo']:.0f}, {r['M_T_hi']:.0f}]",
             f"{r['ipw_empirical_diff']:+.0f} [{r['ipw_empirical_lo']:+.0f}, {r['ipw_empirical_hi']:+.0f}] / "
             f"{r['ipw_model_diff']:+.0f} [{r['ipw_model_lo']:+.0f}, {r['ipw_model_hi']:+.0f}]",
             f"{r['oe_2plus_vs_P']:.2f} / {r['oe_2plus_vs_MT']:.2f}",
             f"{r['D']:+.4f} [{r['D_boot_lo']:+.4f}, {r['D_boot_hi']:+.4f}]",
             f"{r['D_residual']:+.4f} [{r['D_residual_lo']:+.4f}, {r['D_residual_hi']:+.4f}]"]
            for r in read(tables, "S6_concatenation_bias").iter_rows(named=True)]
    t["S6"] = table_html(["Library", "Index molecules of 4 kb or more", "Partner mean (bp)", "Null P (bp)", "M_T (bp)",
                          "After reweighting by 1/w(T): gap to independence (bp) [95% CI], empirical / model w(T)",
                          "Arrays with 2+ molecules of 4 kb or more: obs/exp against P / M_T",
                          "D [bootstrap 95% CI]", "Residual D beyond M_T [95% interval]"], rows, "num")

    hifi = {x["short"]: x["hifi_analysed"] for x in lib}
    share = {(r["library"], r["pattern"]): r["share"] for r in read(tables, "fig2a_chain_patterns").iter_rows(named=True)}
    art = {(r["library"], r["class"]): r["per_1000_molecules"] for r in read(tables, "fig2b_artifact_rates").iter_rows(named=True)}
    bdf = {r["library"]: r for r in read(tables, "S7_bd144_parent_read_equivalents").iter_rows(named=True)}
    af = read(tables, "S7_argentag_parent_reads").row(0, named=True)
    rows = [
        ["HiFi reads analysed", *[fmt(hifi[x]) for x in ORDER3], fmt(af["hifi_zmws"])],
        ["Mirror reads, share of HiFi reads", *[pct(share[(x, "mirror")], 3) for x in ORDER3], pct(af["mirror_share"], 3)],
        ["Mirror junctions holding a TSO-TSO molecule", *[pct(bdf[x]["junctions_tso_tso_share"], 1) for x in ORDER3], pct(af["junctions_tso_tso_share"], 1)],
        ["TSO-TSO junctions at adapter A", *[fmt(bdf[x]["junctions_at_A_tso_tso"]) for x in ORDER3], fmt(int(af["junctions_at_A_tso_tso"]))],
        ["Single TSO-TSO junction reads with one molecule per slot from X to Q on each of 2 arms plus 1 at the junction", *[pct(bdf[x]["single_tt_matching_rule"], 1) for x in ORDER3],
         pct(af["single_tt_matching_rule"], 1)],
        ["TSO-TSO molecules per 1,000 molecules", *[f"{art[(x, 'TSO-TSO')]:.3g}" for x in ORDER3], f"{af['tso_tso_per_1000']:.3g}"],
        ["R1-R1 molecules per 1,000 molecules", *[f"{art[(x, 'R1-R1')]:.3g}" for x in ORDER3], f"{af['r1_r1_per_1000']:.3g}"],
        ["Usable molecules per mirror read", *[f"{bdf[x]['usable_per_mirror_read']:.2f}" for x in ORDER3], f"{af['usable_per_mirror_read']:.2f}"],
        ["Mirror shortfall, molecules per HiFi read", *[f"{bdf[x]['mirror_shortfall']:.3f}" for x in ORDER3], f"{af['mirror_shortfall_ours_per_hifi_zmw']:.3f}"],
    ]
    t["S7"] = table_html(["Full-read readout", "BD144 ND", "BD144 HM", "BD144 kin", "ArgenTag"], rows, "num")

    rows = [[x if x == ARG else f"BD144 {x}", *(f(n[f"{KEY[x]}_{k}"]) for k, f in (
        ("true_mirror", lambda v: pct(v, 3)), ("q_not_a", lambda v: pct(v, 2)), ("q_not_a_ppv", lambda v: f"{v:.3f}"), ("flag", lambda v: pct(v, 3)),
        ("flag_ppv", lambda v: f"{v:.3f}"), ("flag_sensitivity", lambda v: f"{v:.3f}"), ("flag_non_mirror", lambda v: pct(v, 3))))]
            for x in ORDER3 + [ARG]]
    t["S8"] = table_html(["Library", "Mirror reads (parent reads)", "Q present, A absent", "PPV for a mirror read", "Unwritten-head flag",
                          "PPV", "Sensitivity", "Non-mirror reads flagged"], rows, "num")

    hv = {("1.2.0" if "1.2.0" in r["skera"] else "26.1.0"): r for r in read(tables, "S9_skera_version_check").iter_rows(named=True)}
    rows = [[f"skera {v}", fmt(hv[v]["sreads"]), pct(hv[v]["q_not_a_share"], 2), pct(hv[v]["flagged_share"], 2), f"{hv[v]['ppv']:.3f}",
             f"{hv[v]['sensitivity']:.3f}"] for v in ("1.2.0", "26.1.0")]
    t["S9"] = table_html([f"Same {fmt(hv['1.2.0']['hifi_zmws'])} ND HiFi reads", "S-reads", "Q present, A absent", "Unwritten-head flag", "PPV",
                          "Sensitivity"], rows, "num")

    rows = [[label, fmt(n[f"mirror_{reg}_molecules"]),
             *[pct(n[f"mirror_{reg}_{c}_share"], 1) if f"mirror_{reg}_{c}_share" in n else "0%" for c in MIRROR_CLASSES]]
            for reg, label in MIRROR_REGIONS.items()]
    t["S10"] = table_html(["Proper molecules (ND)", "Molecules", "Clean S-read", "Fused with a neighbour", "Partly covered", "Not written"], rows, "num")
    return t


def build_figures(tables: Path, numbers: dict, reads: list[dict]) -> dict[str, go.Figure]:
    """Supplementary Figs. 2 to 9 as plotly figures; Fig. 3 draws the example reads of fig3_reads.json."""
    d = figures.load(tables, numbers)
    return {k: figures.fig3(reads) if k == "fig3" else getattr(figures, k)(d) for k in FIGURES}


def schematics(numbers: dict) -> dict[str, str]:
    """The two drawings by slot: Fig. 1 with kin's lima pass rates of chimeras and fragments, Fig. 4a."""
    kin = {"fates": {"fused_lima": numbers["kin_false_cut_fused_lima"], "frag_lima": numbers["kin_false_cut_frag_lima"]}}
    return {"fig1": false_cut_svg(kin), "fig4a": schematic_svg()}


def plot_div(fig: go.Figure, name: str) -> str:
    div = pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=f"{name}-plot", default_width="100%", default_height="100%",
                      config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]})     # figures/ holds SVG and PDF
    return f'<div class="plotwrap">{div}</div>'


def font_faces() -> str:
    """A <style> of @font-face rules, one per FONTS file inlined as base64, so the page fetches no font."""
    rules = "".join(f'@font-face {{ font-family: "{family}"; font-style: {style}; font-weight: {weight}; '
                    f'src: url(data:font/woff2;base64,{base64.b64encode((FONT_DIR / name).read_bytes()).decode()}) format("woff2"); }}\n'
                    for family, weight, style, name in FONTS)
    return f"<style>\n{rules}</style>"


def note_html(tables: Path, numbers: dict, reads: list[dict], variant: str = "screen", figs: dict[str, go.Figure] | None = None) -> str:
    """The Note as one HTML page with its fonts inlined. screen: plotly figures (figs, built from tables/ and reads when not
    given) with the plotly bundle inlined once; print: the figures as figures/Supplementary_<fig>.svg images, and print.css."""
    page = fill(TEMPLATE.read_text(), numbers)
    title = re.search(r"<h1>(.*?)</h1>", page).group(1)
    page = page.replace("<!--title-->", title, 1).replace("<!--fonts-->", font_faces(), 1)
    for name, html in supplementary_tables(tables, numbers).items():
        page = page.replace(f"<!--table {name}-->", html, 1)
    for slot, svg in schematics(numbers).items():
        page = page.replace(f"<!--schematic {slot}-->", svg, 1)
    if variant == "print":
        for k in FIGURES:
            page = page.replace(f"<!--figure {k}-->", f'<img class="printfig" src="figures/Supplementary_{k}.svg" alt="Supplementary Fig. {k[3:]}">', 1)
        return page.replace("</head>", f"<style>\n{PRINT_CSS.read_text()}</style>\n</head>", 1)
    figs = figs or build_figures(tables, numbers, reads)
    for k in FIGURES:
        page = page.replace(f"<!--figure {k}-->", plot_div(figs[k], k), 1)
    js = plotly.offline.get_plotlyjs().replace("\ufffd", "\\ufffd")   # the bundle's literal U+FFFD sits in JS strings; escape it
    return page.replace("<main>", f"<script>{js}</script>\n<main>", 1)


def write_schematics(svgs: dict[str, str], out: Path) -> None:
    """Each drawing as <name>.svg and, through Inkscape, <name>.pdf. Without Inkscape the PDFs are skipped, with a message."""
    inkscape = shutil.which("inkscape")
    if inkscape:
        version = subprocess.run([inkscape, "--version"], capture_output=True, text=True).stdout.strip()
        print(f"render.py: drawings to PDF with {version}")
    for name, svg in svgs.items():
        (out / f"{name}.svg").write_text(svg)
        if inkscape:
            subprocess.run([inkscape, str(out / f"{name}.svg"), "--export-type=pdf", f"--export-filename={out / name}.pdf"], check=True, capture_output=True)
    if not inkscape:
        print(f"render.py: inkscape not found; not written: {', '.join(f'{name}.pdf' for name in svgs)} (the SVGs are)", file=sys.stderr)


def write_figures(figs: dict[str, go.Figure], out: Path) -> None:
    pio.kaleido.scope.mathjax = None                         # otherwise the first PDF carries a "Loading [MathJax]" box
    for k, f in figs.items():
        for ext in ("svg", "pdf"):
            f.write_image(str(out / f"Supplementary_{k}.{ext}"), width=f.layout.width or 1000, height=f.layout.height or 500)


def print_pdf(page: Path, pdf: Path) -> None:
    """page printed to pdf by headless Chrome."""
    chrome = shutil.which("google-chrome")
    if not chrome:
        sys.exit(f"render.py: google-chrome not found; {pdf} not written (print {page} to PDF by hand)")
    r = subprocess.run([chrome, "--headless=new", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", page.resolve().as_uri()],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not pdf.is_file():
        sys.exit(f"render.py: google-chrome failed to print {page} (exit {r.returncode}):\n{r.stderr}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", required=True, help="the report folder compute.py wrote: tables/, numbers.json, fig3_reads.json")
    report = Path(ap.parse_args(argv).report_dir)
    tables = report / "tables"
    numbers = json.loads((report / "numbers.json").read_text())
    reads = json.loads((report / "fig3_reads.json").read_text())
    check_claims(tables, numbers)
    figs = build_figures(tables, numbers, reads)
    out = report / "figures"
    out.mkdir(exist_ok=True)
    write_schematics({SCHEMATICS[slot]: svg for slot, svg in schematics(numbers).items()}, out)
    write_figures(figs, out)
    (report / "supplementary_note.html").write_text(note_html(tables, numbers, reads, figs=figs), encoding="utf-8")
    page = report / "supplementary_note.print.html"
    page.write_text(note_html(tables, numbers, reads, variant="print"), encoding="utf-8")
    print_pdf(page, report / "supplementary_note.pdf")
    print("wrote", report / "supplementary_note.html", report / "supplementary_note.pdf", out)


if __name__ == "__main__":
    main()
