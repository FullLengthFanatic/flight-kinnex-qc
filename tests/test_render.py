"""Tests for study/report/render.py: the template's placeholders and slots, the formats (v10's helpers), the Supplementary
Tables built from the source tables, the text rules, the check of the false-cut claim, and the whole render step (HTML,
print HTML, PDF and figure files) on the golden tables and numbers.json. The numbers of the text are checked against
v10's in tests/test_golden.py."""
import json
import re
import shutil

import plotly
import pytest

from golden_text import visible_text
from pipeline_helpers import EXAMPLE_TRACKS, GOLDEN

import render

TITLE = "Supplementary Note: Kinnex array integrity in FLIGHT-seq libraries"
NUMBERS = json.loads((GOLDEN / "v10_numbers.json").read_text())
TABLES = ["S1", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]      # Supplementary Table 2 is prose: it is in the template
FIGURES = ["fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9"]
# The writing-voice rules: none of these words or phrases in the Note (word starts, case ignored).
BANNED = ("delve leverage harness utilize optimize capitalize streamline unlock empower elevate catalyze foster bolster underscore "
          "showcase garnered embark crucial vital paramount pivotal profound groundbreaking transformative cutting-edge unprecedented "
          "game-changing revolutionary state-of-the-art compelling sophisticated robust seamless innovative tapestry labyrinth crucible "
          "landscape realm fabric journey narrative nuanced multifaceted intricate delighted thrilled resonated genuinely furthermore "
          "moreover additionally consequently ultimately").split() + [
    "woven into", "that said", "it's worth noting", "it bears mentioning", "in conclusion", "is real", "are real", "the whole game",
    "are exactly", "is precisely", "absolutely", "food for thought", "worth unpacking", "useful lens", "actually matters"]


@pytest.fixture(scope="module")
def html() -> str:
    return render.note_html(GOLDEN / "v10_tables", NUMBERS, EXAMPLE_TRACKS)


@pytest.fixture(scope="module")
def print_html() -> str:
    return render.note_html(GOLDEN / "v10_tables", NUMBERS, EXAMPLE_TRACKS, variant="print")


def test_template_uses_only_numbers_keys_and_every_format():
    found = render.placeholders(render.TEMPLATE.read_text())
    assert found
    assert sorted({k for k, _ in found} - set(NUMBERS)) == []
    assert {f for _, f in found} == set(render.FORMATS)


def test_fill_refuses_unknown_keys_and_formats_and_a_key_without_format():
    with pytest.raises(KeyError, match="no_such_key"):
        render.fill("a {no_such_key|pct1} b", NUMBERS)
    with pytest.raises(KeyError, match="pct9"):
        render.fill("{nd_mirror_share|pct9}", NUMBERS)
    with pytest.raises(ValueError, match="nd_mirror_share"):
        render.fill("{nd_mirror_share}", NUMBERS)
    assert render.fill("x {nd_mirror_share|pct1} .a { color: red; } .b{fill:#FFF}", NUMBERS) == "x 16.1% .a { color: red; } .b{fill:#FFF}"


def test_formats_are_v10s_helpers():
    f = render.FORMATS
    assert (f["fmt"](1234567), f["fmt"](6.7213), f["int"](8308), f["comma0"](2976.0)) == ("1,234,567", "6.72", "8,308", "2,976")
    assert (f["pct0"](0.18), f["pct1"](0.16105), f["pct2"](0.0014728)) == ("18%", "16.1%", "0.15%")
    assert (f["small_pct"](0.00095472), f["sig2"](207), f["sig2"](2978), f["pp1"](0.000615)) == ("0.095%", "210", "3,000", "0.1")
    assert (f["f0"](891.0), f["f1"](7.5554), f["f2"](0.4), f["f3"](0.0039224)) == ("891", "7.6", "0.40", "0.004")
    assert (f["sf1"](-2.507), f["sf3"](-0.00464), f["sf4"](0.000269), f["g2"](0.10917)) == ("-2.5", "-0.005", "+0.0003", "0.11")
    assert (f["kb1"](6628.0), f["mio2"](1969703), f["str"]("0.1.0")) == ("6.6", "1.97", "0.1.0")
    assert (f["list"]([986295, 1051947]), f["commits"](["5e988fe"]), f["commits"](["a", "b", "c"])) == ("986295, 1051947", "5e988fe", "a, b and c")


def test_title_in_one_place_of_the_template(html):
    assert render.TEMPLATE.read_text().count(TITLE) == 1
    assert f"<title>{TITLE}</title>" in html and f"<h1>{TITLE}</h1>" in html


def test_literals_that_equal_computed_values_are_placeholders():
    tpl = render.TEMPLATE.read_text()
    assert "99.1%" not in tpl and "5,000" not in tpl
    assert "In a test on ND, {downstream_unwritten_mapped|pct1} of such molecules map." in tpl
    assert "re-called mirror reads in {arg_independent_sample|fmt} random reads" in tpl
    assert "takes\nthe first {tso_both_ends_checked|fmt} skera 1.2.0 S-reads" in tpl


def test_every_table_and_figure_has_one_slot_in_order():
    tpl = render.TEMPLATE.read_text()
    assert re.findall(r"<!--table (\w+)-->", tpl) == TABLES
    assert re.findall(r"<!--figure (\w+)-->", tpl) == FIGURES
    assert re.findall(r"<!--schematic (\w+)-->", tpl) == ["fig1", "fig4a"]


def test_supplementary_tables_from_the_source_tables():
    t = render.supplementary_tables(GOLDEN / "v10_tables", NUMBERS)
    assert list(t) == TABLES
    assert {k: v.count("<tr>") - 1 for k, v in t.items()} == {"S1": 4, "S3": 3, "S4": 9, "S5": 4, "S6": 3, "S7": 9, "S8": 4, "S9": 2, "S10": 4}
    # rows as v10 printed them
    assert ("<tr><td>BD144 ND</td><td>PB538_3_r0912</td><td>None</td><td>10,945,011</td><td>1,969,703</td><td>13,725,691</td>"
            "<td>6.97 / 6.97</td><td>81.4% / 81.4%</td></tr>") in t["S1"]
    assert "<tr><td>Complete arrays</td><td>complete</td><td>100.0%</td><td>100.0%</td><td>99.5%</td><td>67.0%</td></tr>" in t["S4"]
    assert ("<tr><td>ND</td><td>174,460</td><td>1412</td><td>1423 [1421, 1425]</td><td>1411 [1410, 1413]</td>"
            "<td>-3 [-4, -1] / -14 [-15, -12]</td><td>0.94 / 0.99</td><td>-0.0012 [-0.0019, -0.0005]</td>"
            "<td>-0.0011 [-0.0022, -0.0001]</td></tr>") in t["S6"]
    assert "<tr><td>BD144 ND</td><td>16.105%</td><td>14.67%</td><td>0.917</td><td>9.512%</td><td>0.992</td><td>0.586</td><td>0.076%</td></tr>" in t["S8"]
    assert "<tr><td>Arm after the junction (X to Q, forward adapters)</td><td>579,105</td><td>85.9%</td><td>1.9%</td><td>1.0%</td><td>11.2%</td></tr>" in t["S10"]


def test_text_rules(html, print_html):
    """No en or em dash outside the inlined plotly bundle, none of the banned words in the text."""
    bundle = re.search(r"<script>/\*\*\n\* plotly\.js v.*?</script>", html, flags=re.S)
    assert bundle
    for page in (html.replace(bundle.group(0), ""), print_html):
        assert "\u2013" not in page and "\u2014" not in page
    text = visible_text(html).lower()
    assert [w for w in BANNED if re.search(r"\b" + re.escape(w), text)] == []


def test_screen_variant_inlines_plotly_once_and_print_variant_draws_svg_files(html, print_html):
    js = plotly.offline.get_plotlyjs()[:300]
    assert html.count(js) == 1 and html.count('class="plotly-graph-div"') == len(FIGURES)
    assert "<script" not in print_html and "plotly-graph-div" not in print_html
    assert re.findall(r'<img class="printfig" src="figures/Supplementary_(fig\d)\.svg"', print_html) == FIGURES
    assert render.PRINT_CSS.read_text() in print_html and render.PRINT_CSS.read_text() not in html
    assert visible_text(print_html) == visible_text(html)


def test_false_cut_lima_claim_is_checked_against_the_source_table(tmp_path):
    """Fig. 1's caption says the lima pass rates of ND and HM differ from kin's by less than 0.1 percentage point; render
    stops rather than print it when the false-cut anatomy table says otherwise."""
    render.check_claims(GOLDEN / "v10_tables")
    tables = tmp_path / "tables"
    shutil.copytree(GOLDEN / "v10_tables", tables)
    p = tables / "fig1_skera_false_cut_anatomy.tsv"
    p.write_text(p.read_text().replace("\t0.004327740762153879\t", "\t0.005327740762153879\t"))
    with pytest.raises(SystemExit, match="0.1 percentage point"):
        render.check_claims(tables)


def test_schematic_pdfs_skipped_with_a_message_without_inkscape(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    render.write_schematics({"Supplementary_fig1": '<svg xmlns="http://www.w3.org/2000/svg"/>'}, tmp_path)
    assert (tmp_path / "Supplementary_fig1.svg").is_file() and not (tmp_path / "Supplementary_fig1.pdf").exists()
    err = capsys.readouterr().err
    assert "inkscape not found" in err and "Supplementary_fig1.pdf" in err


def test_pdf_needs_chrome(tmp_path, monkeypatch):
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="google-chrome"):
        render.print_pdf(tmp_path / "supplementary_note.print.html", tmp_path / "supplementary_note.pdf")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """render.py --report-dir on a report folder holding only what compute writes: tables/, numbers.json, fig3_reads.json."""
    if not shutil.which("google-chrome"):
        pytest.skip("google-chrome is not installed")
    report = tmp_path_factory.mktemp("report")
    shutil.copytree(GOLDEN / "v10_tables", report / "tables")
    shutil.copy(GOLDEN / "v10_numbers.json", report / "numbers.json")
    (report / "fig3_reads.json").write_text(json.dumps(EXAMPLE_TRACKS))
    render.main(["--report-dir", str(report)])
    return report


def test_render_writes_the_note_print_variant_pdf_and_figures(rendered):
    assert sorted(p.name for p in rendered.iterdir()) == ["fig3_reads.json", "figures", "numbers.json", "supplementary_note.html",
                                                          "supplementary_note.pdf", "supplementary_note.print.html", "tables"]
    assert (rendered / "supplementary_note.pdf").read_bytes()[:5] == b"%PDF-"
    figs = {p.name for p in (rendered / "figures").iterdir()}
    plots = {f"Supplementary_{k}.{ext}" for k in FIGURES for ext in ("svg", "pdf")}
    drawings = {"Supplementary_fig1.svg", "Supplementary_fig4a_schematic.svg"}
    if shutil.which("inkscape"):
        drawings |= {"Supplementary_fig1.pdf", "Supplementary_fig4a_schematic.pdf"}
    assert figs == plots | drawings
    for name in figs:
        head = (rendered / "figures" / name).read_bytes()[:200]
        assert head.startswith(b"%PDF-") if name.endswith(".pdf") else b"<svg" in head
    printed = (rendered / "supplementary_note.print.html").read_text()
    assert all((rendered / src).is_file() for src in re.findall(r'<img class="printfig" src="([^"]+)"', printed))
