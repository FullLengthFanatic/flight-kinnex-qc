"""Tests for study/report/render.py: the template's placeholders and slots, the formats (v10's helpers) and ranges, the
Supplementary Tables built from the source tables, the text rules, the checks of the claims that hold only for these data,
the inlined fonts, and the whole render step (HTML, print HTML, PDF and figure files) on the golden tables and numbers.json. The numbers of the text are checked against
v10's in tests/test_golden.py."""
import base64
import json
import re
import shutil
import subprocess

import plotly
import pytest

from golden_text import visible_text
from pipeline_helpers import EXAMPLE_TRACKS, GOLDEN

import render

TITLE = "Supplementary Note: Kinnex array integrity in FLIGHT-seq libraries"
NUMBERS = json.loads((GOLDEN / "v10_numbers.json").read_text())
TABLES = ["S1", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]      # Supplementary Table 2 is prose: it is in the template
FIGURES = ["fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9"]
# The writing-voice rules: none of these words or phrases in the Note, every category of the voice file's banned list
# (trailing ":" and "..." of its scaffolding phrases dropped), plus "actually matters". Case is ignored; a single word
# also matches as a word start (leveraged, robustly), a phrase only as whole words.
BANNED = {
    "verbs": ["delve", "leverage", "harness", "utilize", "optimize", "capitalize on", "streamline", "unlock", "empower", "elevate", "catalyze",
              "foster", "bolster", "underscore", "showcase", "garnered", "embark"],
    "intensifiers": ["crucial", "vital", "paramount", "pivotal", "profound", "groundbreaking", "transformative", "cutting-edge", "unprecedented",
                     "game-changing", "revolutionary", "state-of-the-art", "compelling", "sophisticated", "robust", "seamless", "innovative"],
    "abstract": ["tapestry", "labyrinth", "crucible", "landscape", "realm", "fabric", "woven into", "journey", "narrative", "nuanced",
                 "multifaceted", "intricate"],
    "stock openings": ["i hope this email finds you well", "i wanted to reach out", "thanks for sharing this", "great question"],
    "hollow enthusiasm": ["i'd be happy to", "delighted", "thrilled", "please don't hesitate to reach out", "resonated", "struck by",
                          "that's exactly", "keeps me up at night", "i love this", "this is fantastic", "what a great question", "absolutely", "100%"],
    "faux-depth closers": ["the conversation is just beginning", "there's more to unpack here", "food for thought",
                           "let's keep the conversation going", "worth sitting with", "worth unpacking"],
    "faux-precision hedges": ["genuinely", "i'd genuinely like to", "i truly think", "i really believe"],
    "metaphors": ["gets less airtime", "useful lens", "compounds that", "sits at the intersection of"],
    "mechanical transitions": ["furthermore", "moreover", "additionally", "consequently", "in conclusion", "ultimately", "that said",
                               "that being said", "it's worth noting", "it bears mentioning"],
    "hollow emphasis": ["is real", "are real", "the whole game", "are exactly", "is precisely", "are the very ones"],
    "performed reactions": ["stopped me cold", "hit me", "struck me", "this is where it gets interesting"],
    "scaffolding": ["the short version", "the bigger point", "step back from", "here's what i think", "here's why"],
    "global rules": ["actually matters"],
}


def banned_hits(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    hits = []
    for phrase in (p for ps in BANNED.values() for p in ps):
        end = (r"(?!\w)" if phrase[-1].isalnum() else "") if " " in phrase else r"\w*"
        hits += [m.group(0) for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + end, text)]
    return hits


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


def test_a_range_prints_one_value_when_both_ends_print_the_same():
    """{lo..hi|format} is v10's span(): "lo to hi", or a single value when both ends print the same in that format."""
    n = {"a": 0.0861, "b": 0.0878, "c": 0.0864}
    assert render.fill("{a..b|pct1}", n) == "8.6% to 8.8%"
    assert render.fill("{a..c|pct1}", n) == "8.6%"
    assert render.fill("{a..c|pct2}", n) == "8.61% to 8.64%"
    with pytest.raises(KeyError, match="no_such_key"):
        render.fill("{a..no_such_key|pct1}", n)
    assert render.placeholders("{a..b|pct2} and {c|pct1}") == [("a", "pct2"), ("b", "pct2"), ("c", "pct1")]


def test_ranges_where_v10_used_span():
    """v10 printed five ranges with span() (the shares of reads where our segmentation misses an adapter, in section 1 and
    Supplementary Table 2, and the lima pass rates of those reads); the template prints them as ranges, and only them."""
    found = re.findall(r"\{([a-z0-9_]+)\.\.([a-z0-9_]+)\|([a-z0-9_]+)\}", render.TEMPLATE.read_text())
    assert [(lo.removesuffix("_min"), hi.removesuffix("_max"), f) for lo, hi, f in found] == [
        ("missed_adapter_share", "missed_adapter_share", "pct2"), ("sread_lima_pass_missed", "sread_lima_pass_missed", "pct1"),
        ("sread_lima_pass_complete", "sread_lima_pass_complete", "pct1"), ("missed_adapter_share", "missed_adapter_share", "pct2"),
        ("skipped_slot_share", "skipped_slot_share", "pct2")]
    assert all(lo.endswith("_min") and hi.endswith("_max") for lo, hi, _ in found)


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
    assert "99.1%" not in tpl and "5,000" not in tpl and "200,000" not in tpl
    assert tpl.count("{cal_hifi_reads|fmt} ND HiFi reads") == 4          # terms, section 1, Supplementary Table 9's title, Methods
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
    assert banned_hits(visible_text(html)) == []


def test_banned_words_are_found_in_any_case_and_form():
    assert banned_hits("A ROBUST call. They Leveraged it; 100% sure. That said, it Hit me.") == ["leveraged", "robust", "100%", "that said", "hit me"]
    assert banned_hits("The adapter hit means nothing; it is realised; 100.0% of reads; 2100%; vitamins aside.") == []
    assert banned_hits("Here’s why.") == ["here's why"]


def test_screen_variant_inlines_plotly_once_and_print_variant_draws_svg_files(html, print_html):
    js = plotly.offline.get_plotlyjs()[:300]
    assert html.count(js) == 1 and html.count('class="plotly-graph-div"') == len(FIGURES)
    assert "<script" not in print_html and "plotly-graph-div" not in print_html
    assert re.findall(r'<img class="printfig" src="figures/Supplementary_(fig\d)\.svg"', print_html) == FIGURES
    assert render.PRINT_CSS.read_text() in print_html and render.PRINT_CSS.read_text() not in html
    assert visible_text(print_html) == visible_text(html)


def test_screen_and_print_pages_load_nothing_from_the_network(html, print_html):
    """No web fonts, stylesheets, scripts or images fetched by URL, so Chrome prints the same PDF offline; hyperlinks (<a href>)
    are allowed. The inlined plotly bundle is left out: its code holds URLs it does not load here."""
    bundle = re.search(r"<script>/\*\*\n\* plotly\.js v.*?</script>", html, flags=re.S)
    for page in (html.replace(bundle.group(0), ""), print_html):
        tags = re.findall(r"<(?!a\b)[a-z][^>]*>", page, flags=re.I)
        assert [t for t in tags if re.search(r"\b(?:src|href)\s*=\s*[\"']?(?:https?:)?//", t, flags=re.I)] == []
        assert not re.search(r"@import|url\(\s*[\"']?(?:https?:)?//", page, flags=re.I)


# The fonts v10 loaded from Google Fonts, as the Note embeds them: family, weight, style and file in resources/fonts.
# Source Serif 4 in its static optical-size cuts: Small Text for the text, Subhead for the title.
FONTS = {("IBM Plex Sans", 400, "normal"): "IBMPlexSans-Regular.woff2", ("IBM Plex Sans", 500, "normal"): "IBMPlexSans-Medium.woff2",
         ("IBM Plex Sans", 600, "normal"): "IBMPlexSans-SemiBold.woff2", ("IBM Plex Mono", 400, "normal"): "IBMPlexMono-Regular.woff2",
         ("IBM Plex Mono", 500, "normal"): "IBMPlexMono-Medium.woff2", ("Source Serif 4", 400, "normal"): "SourceSerif4SmText-Regular.ttf.woff2",
         ("Source Serif 4", 600, "normal"): "SourceSerif4SmText-Semibold.ttf.woff2", ("Source Serif 4", 400, "italic"): "SourceSerif4SmText-It.ttf.woff2",
         ("Source Serif 4 Subhead", 400, "normal"): "SourceSerif4Subhead-Regular.ttf.woff2",
         ("Source Serif 4 Subhead", 600, "normal"): "SourceSerif4Subhead-Semibold.ttf.woff2"}
FONT_DIR = render.HERE.parent.parent / "resources" / "fonts"


def test_screen_and_print_pages_embed_the_fonts(html, print_html):
    """Each font file inlined once as a base64 @font-face rule in both pages; the title is set in the Subhead cut."""
    face = re.compile(r'@font-face \{ font-family: "([^"]+)"; font-style: (\w+); font-weight: (\d+); '
                      r'src: url\(data:font/woff2;base64,([A-Za-z0-9+/=]+)\) format\("woff2"\); \}')
    for page in (html, print_html):
        found = {(fam, int(weight), style): base64.b64decode(data) for fam, style, weight, data in face.findall(page)}
        assert len(face.findall(page)) == len(FONTS)
        assert {k: v == (FONT_DIR / FONTS[k]).read_bytes() for k, v in found.items()} == {k: True for k in FONTS}
    assert re.search(r'h1 \{ font-family: "Source Serif 4 Subhead", var\(--serif\);', html)


def test_fonts_are_kept_with_their_licenses():
    assert sorted(p.name for p in FONT_DIR.iterdir()) == sorted([*FONTS.values(), "OFL-IBM-Plex.txt", "OFL-Source-Serif.md", "README.md"])
    for lic in ("OFL-IBM-Plex.txt", "OFL-Source-Serif.md"):
        assert "SIL Open Font License, Version 1.1" in (FONT_DIR / lic).read_text()
    readme = (FONT_DIR / "README.md").read_text()
    assert all(f"`{name}`" in readme for name in FONTS.values())
    assert all(v in readme for v in ("`@ibm/plex-sans@1.1.0`", "`@ibm/plex-mono@2.5.0`", "`4.005R`"))


# The template's words that hold only for these data, each with a change to the source tables (table, old text, new text)
# or to numbers.json (key, value) that makes it false.
CLAIMS = [
    ("ND and HM differ by less than 0.1 percentage point",
     ("fig1_skera_false_cut_anatomy", "\t0.004327740762153879\t", "\t0.005327740762153879\t")),
    ("under 1% of the mean partner", ("S6_concatenation_bias", "\t-11.068129616285432\t", "\t-15.0\t")),
    ("never calls a read a mirror array when it is not one",
     ("S4_chain_classification_benchmark_summary", "0.01\t0.998125\t1.0\t0\t1200", "0.01\t0.998125\t0.999\t1\t1200")),
    ("none of them at adapter A", ("S7_argentag_parent_reads", "\t0.0004591255298390601\t0\t", "\t0.0004591255298390601\t2\t")),
    ("junctions occur at adapters B to Q but not at A", ("fig4b_mirror_junction_letters_ND", "A\tother\t13\n", "A\tother\t13\nA\tTSO-TSO\t1\n")),
    ("with junctions at A to H but never at Q", ("fig4b_mirror_junction_letters_ND", "Q\tother\t56", "Q\tother\t56\nQ\tR1-R1\t1")),
    ("in all three libraries alike", ("false_cut_reads_share_max", 0.096)),
    ("a slight underestimate", ("nd_tso_tso_only_skera_pass_share", 0.011)),
]


def test_passages_a_reader_could_not_follow_are_rewritten_and_listed(html):
    """The Note's readability edits: what the unwritten molecules map as often as (second key finding), which molecules the
    end classes of Supplementary Fig. 2b count, what fail reads and the bead linkers are (Methods). CHANGELOG.md lists them
    in a line under its entry on the Note's text."""
    text = visible_text(html)
    for s in ("skera leaves many molecules of mirror arrays unwritten (observed), and those molecules map as often as molecules from complete reads.",
              "The end classes count only molecules with no primer inside them; fused molecules are counted separately, whatever their ends.",
              "For fail reads (reads that failed to reach HiFi quality), the TSO threshold was relaxed to 4 edits",
              "The two fixed linker sequences between the three parts of the BD Rhapsody bead's cell label (GTGA and GACA) and an anchored polyT "
              "were scored after Read1."):
        assert s in text, s
    assert "they map as well as any" not in text and "bead cell label linkers" not in text
    changelog = (render.HERE.parent.parent / "CHANGELOG.md").read_text().splitlines()
    note_entry = next(i for i, line in enumerate(changelog) if line.startswith("- The Supplementary Note is built by"))
    assert changelog[note_entry + 1].startswith("  - Readability edits")


def test_passages_a_reader_could_not_follow_in_round_2_are_rewritten_and_listed(html):
    """Round 2 of the Note's readability edits: the mirror junction defined in the first key finding; how adapters join (Terms);
    the tenths of the sample, the chimera's real adapter, the reverse-complement reads, the CTCCTCC repeat and why every real
    adapter sits next to a TSO or Read1 (section 1); the mirror size rule in words in sections 3 and 6 and Supplementary Table
    7, with the formula of the Fig. 4c plot explained in its legend. No number token changes (tests/test_golden.py); CHANGELOG.md
    lists the edits in the line after round 1's, and the Note tracked in report/ is rebuilt with them."""
    text = visible_text(html)
    tracked = visible_text((render.HERE.parent.parent / "report" / "supplementary_note.html").read_text(encoding="utf-8"))
    new = ["of their mirror junctions (an adapter followed by its own reverse complement) hold a TSO-TSO molecule.",
           "An adapter can join only the same adapter on another molecule (the B end of a molecule from the first slot joins the B end of "
           "one from the second), which keeps the slots in this order.",
           "across different tenths of the sample.",
           "a whole molecule, then the real adapter skera did not cut at, then the first part of the cut molecule.",
           "on reads that cover the array's reverse complement (adapters Q to A) it keeps the Read1 end",
           "CTCCTCC repeat sequence appears within 10 bp of", "of the false-cut sites, so the look-alike sequences are not one fixed motif.",
           "next to a TSO or Read1, and so rejects the look-alikes inside cDNA: every real adapter lies between two molecules (or at an end "
           "of the array), so it sits next to a molecule end.",
           "holds one molecule per slot from X to Q on each of its 2 arms, plus 1 TSO-TSO molecule (five in all with the junction at G, "
           "whose arms each span G to H and H to Q), a count that",
           "hold exactly the predicted number of molecules, one per slot from X to Q on each of the 2 arms plus 1 TSO-TSO molecule.",
           "against the prediction 2 x (slots from X to Q) + 1 molecules: one per slot from the junction adapter X to Q on each arm, plus the "
           "TSO-TSO molecule (five with the junction at G).",
           " Single TSO-TSO junction reads with one molecule per slot from X to Q on each of 2 arms plus 1 at the junction "]
    assert [s for s in new if s not in text] == [] and [s for s in new if s not in tracked] == []
    old = ["of their junctions hold", "an adapter joins only its own letter", "between tenths of the sample", "the skipped adapter and the first part",
           "reads from the other strand (Q to A)", "so the false matches", "where every real adapter sits, and so rejects"]
    assert [s for s in old if s in text] == [] and [s for s in old if s in tracked] == []
    assert text.count("2 x (slots from X to Q) + 1") == 1 == tracked.count("2 x (slots from X to Q) + 1")
    changelog = (render.HERE.parent.parent / "CHANGELOG.md").read_text().splitlines()
    note_entry = next(i for i, line in enumerate(changelog) if line.startswith("- The Supplementary Note is built by"))
    assert changelog[note_entry + 2].startswith("  - Readability edits, second round")


def test_claims_hold_for_the_golden_data(html):
    render.check_claims(GOLDEN / "v10_tables", NUMBERS)
    assert [claim for claim, _ in CLAIMS if claim not in visible_text(html)] == []


@pytest.mark.parametrize("claim, change", CLAIMS, ids=[c for c, _ in CLAIMS])
def test_claims_are_checked_against_the_data(tmp_path, claim, change):
    """render stops with one line naming the claim rather than print it when the tables or numbers.json say otherwise."""
    tables, numbers = tmp_path / "tables", dict(NUMBERS)
    shutil.copytree(GOLDEN / "v10_tables", tables)
    if len(change) == 3:
        p = tables / f"{change[0]}.tsv"
        assert p.read_text().count(change[1]) == 1
        p.write_text(p.read_text().replace(change[1], change[2]))
    else:
        numbers[change[0]] = change[1]
    with pytest.raises(SystemExit) as e:
        render.check_claims(tables, numbers)
    assert claim in str(e.value) and "\n" not in str(e.value)


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


def test_pdf_embeds_the_note_fonts(rendered):
    """The PDF carries the text's fonts by name (pdffonts): IBM Plex Sans and Mono, Source Serif 4 Small Text (text, bold run-in
    heads, italic) and Subhead (title). The figures are SVG images, whose text Chrome sets in the machine's fonts."""
    if not shutil.which("pdffonts"):
        pytest.skip("pdffonts (poppler-utils) is not installed")
    out = subprocess.run(["pdffonts", str(rendered / "supplementary_note.pdf")], capture_output=True, text=True, check=True).stdout
    names = {line.split()[0].split("+")[-1] for line in out.splitlines()[2:]}
    assert {"IBMPlexSans", "IBMPlexSans-SmBld", "IBMPlexMono", "SourceSerif4SmText-Regular", "SourceSerif4SmText-Semibold",
            "SourceSerif4SmText-It", "SourceSerif4Subhead-Semibold"} <= names
