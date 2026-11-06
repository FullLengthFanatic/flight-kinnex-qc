"""Golden regression of the Supplementary Note: rendered from the frozen v10 source tables (tests/golden/v10_tables) and the
numbers.json compute writes from the v10 inputs (tests/golden/v10_numbers.json), the numbers a reader sees are v10's
(tests/golden/v10_text_numbers.txt) apart from the changes listed in tests/golden/allowed_changes.txt, and the figures
and tables are first cited in order. Fig. 3 draws per-read data, which is not frozen: the test draws two made-up reads
(its plot is not visible text) and checks the caption's ZMWs, which come from numbers.json.
The Note built from the full study rerun and tracked in report/ is held to the same files, apart from the approved number
changes listed in tests/golden/approved_number_changes.json (only these report/ tests and the real downstream test in
tests/test_downstream_test.py apply them): its tables are v10's
byte for byte but for the listed cells, its numbers.json is v10's but for the keys that name the run and the code and the
listed keys, and its text numbers are v10's up to allowed_changes.txt and the listed tokens. Every listed change must
occur in report/."""
import difflib
import json
import re
import subprocess

import pytest

from golden_text import numbers, visible_text
from pipeline_helpers import APPROVED, EXAMPLE_TRACKS, GOLDEN, REPO, approved_changes, with_approved_cells

import render

V10_TOKENS = GOLDEN / "v10_text_numbers.txt"
ALLOWED = GOLDEN / "allowed_changes.txt"
REPORT = REPO / "report"
# What report/ tracks: the Note, its figures, source tables and numbers, and the track data of the four reads of
# Supplementary Fig. 3, which render needs to rebuild the Note from a fresh clone. No other per-read file. render also
# writes supplementary_note.print.html, the page the PDF is printed from, which .gitignore leaves out.
REPORT_FILES = ["fig3_reads.json", "figures", "numbers.json", "supplementary_note.html", "supplementary_note.pdf", "tables"]
# What render writes into figures/: each plotly figure and each drawing, as SVG and as PDF.
FIGURE_FILES = sorted(f"{name}.{ext}" for name in [*(f"Supplementary_{k}" for k in render.FIGURES), *render.SCHEMATICS.values()]
                      for ext in ("svg", "pdf"))
# The numbers.json keys that name the code and the run, not a result: the package version, the commit compute ran at and
# the commits of the run's step markers.
RUN_KEYS = {"kinnex_qc_version", "kinnex_qc_commit", "run_commits"}


@pytest.fixture(scope="module")
def text() -> str:
    html = render.note_html(GOLDEN / "v10_tables", json.loads((GOLDEN / "v10_numbers.json").read_text()), EXAMPLE_TRACKS)
    return visible_text(html)


def allowed_changes() -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """allowed_changes.txt: one line per change, old tokens, new tokens and the reason, tab-separated; tokens separated
    by spaces, "none" for no token (an insertion or a removal); lines starting with # are comments."""
    rows = []
    for line in ALLOWED.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        old, new, reason = line.split("\t")
        assert reason.strip(), line
        rows.append(tuple(tuple() if x == "none" else tuple(x.split(" ")) for x in (old, new)))
    return rows


def test_text_numbers_are_v10s_apart_from_the_allowed_changes(text):
    v10, now = V10_TOKENS.read_text().split(), numbers(text)
    assert len(v10) == 881
    ops = difflib.SequenceMatcher(None, v10, now, autojunk=False).get_opcodes()
    assert [(tuple(v10[i1:i2]), tuple(now[j1:j2])) for op, i1, i2, j1, j2 in ops if op != "equal"] == allowed_changes()


def test_fig3_caption_names_the_example_zmws(text):
    assert "ZMWs: 986295, 1051947, 1051859, 1379616. Hover for coordinates" in text


def test_figures_and_tables_first_cited_in_order(text):
    """As in v10: the first mention of each Supplementary Fig. 1 to 9 and Table 1 to 10 (in the text, a caption or a
    figure title) comes after the first mention of the one before it."""
    for kind, n in (("Fig. ", 9), ("Table ", 10)):
        first = []
        for i in range(1, n + 1):
            m = re.search(rf"Supplementary {re.escape(kind)}{i}(?!\d)", text)
            assert m, f"Supplementary {kind}{i} is never cited"
            first.append(m.start())
        assert first == sorted(first), kind


def section8_commits(n: dict) -> str:
    """The sentence of section 8 that prints run_commits and kinnex_qc_commit, filled from n."""
    sentence = re.search(r"[^.]*\{run_commits\|commits\}[^.]*\{kinnex_qc_commit\|str\}\.", render.TEMPLATE.read_text()).group(0)
    return " ".join(render.fill(sentence, n).split())


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="the checkout has no .git (an unpacked source archive), so git ls-files cannot list report/")
def test_report_holds_the_note_built_from_the_full_rerun():
    """The files git tracks under report/, so rebuilding the Note in place (render.py --report-dir report) keeps it green."""
    tracked = subprocess.run(["git", "ls-files", "report"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    assert sorted({p.split("/")[1] for p in tracked}) == REPORT_FILES
    assert sorted(p.removeprefix("report/figures/") for p in tracked if p.startswith("report/figures/")) == FIGURE_FILES
    assert subprocess.run(["git", "check-ignore", "-q", "report/supplementary_note.print.html"], cwd=REPO).returncode == 0
    assert "report/** -diff linguist-generated=true" in (REPO / ".gitattributes").read_text().splitlines()


def test_report_tables_are_v10s_byte_for_byte_but_the_approved_cells():
    v10 = sorted((GOLDEN / "v10_tables").iterdir())
    assert sorted(p.name for p in (REPORT / "tables").iterdir()) == [p.name for p in v10]
    assert [p.name for p in v10 if (REPORT / "tables" / p.name).read_bytes() != with_approved_cells(p.name, p.read_bytes().decode()).encode()] == []


def test_report_numbers_are_v10s_but_the_run_and_code_keys_and_the_approved_keys():
    built, v10 = (json.loads(p.read_text()) for p in (REPORT / "numbers.json", GOLDEN / "v10_numbers.json"))
    assert list(built) == list(v10)
    approved = v10 | {c["key"]: c["new"] for c in approved_changes("numbers")}
    # json.dumps tells 1 from 1.0
    assert [k for k in v10 if k not in RUN_KEYS and json.dumps(built[k]) != json.dumps(approved[k])] == []
    # the full rerun, and its downstream_test step run again with the best_cover fix
    assert built["run_commits"] == ["7d419a4", "c3f47fb"]


def report_text_numbers() -> list[str]:
    """The text numbers of the Note in report/. allowed_changes.txt is written against the frozen numbers.json, so section
    8's sentence that prints run_commits and kinnex_qc_commit is masked: as report/numbers.json prints it, it is replaced by
    the sentence the frozen numbers print."""
    built, frozen = (json.loads(p.read_text()) for p in (REPORT / "numbers.json", GOLDEN / "v10_numbers.json"))
    text = visible_text((REPORT / "supplementary_note.html").read_text(encoding="utf-8"))
    assert text.count(section8_commits(built)) == 1
    return numbers(text.replace(section8_commits(built), section8_commits(frozen)))


def test_report_note_text_numbers_are_v10s_apart_from_the_allowed_changes_and_the_approved_tokens():
    v10, now = V10_TOKENS.read_text().split(), report_text_numbers()
    for c in approved_changes("text"):
        assert v10[c["token"]] == c["v10"], c
        v10[c["token"]] = c["new"]
    ops = difflib.SequenceMatcher(None, v10, now, autojunk=False).get_opcodes()
    assert [(tuple(v10[i1:i2]), tuple(now[j1:j2])) for op, i1, i2, j1, j2 in ops if op != "equal"] == allowed_changes()


def cell(table, row: str, column: str) -> str:
    """One cell of a TSV table as text, its row named by the row's first column."""
    header, *rows = (line.split("\t") for line in table.read_text().splitlines())
    return next(r[header.index(column)] for r in rows if r[0] == row)


def test_every_approved_change_occurs():
    """No stale entry in approved_number_changes.json: each names a v10 value and a different new value, report/ holds the
    new value where v10 holds the old one, and exactly one line of CHANGELOG.md holds the entry's changelog quote."""
    changes = json.loads(APPROVED.read_text())["changes"]
    assert changes and {c["kind"] for c in changes} <= {"numbers", "table", "text"}
    changelog = (REPO / "CHANGELOG.md").read_text().splitlines()
    for c in changes:
        assert c["v10"] != c["new"] and sum(c["changelog"] in line for line in changelog) == 1, c
    built, v10 = (json.loads(p.read_text()) for p in (REPORT / "numbers.json", GOLDEN / "v10_numbers.json"))
    for c in approved_changes("numbers"):
        assert (json.dumps(v10[c["key"]]), json.dumps(built[c["key"]])) == (json.dumps(c["v10"]), json.dumps(c["new"])), c
    for c in approved_changes("table"):
        assert [cell(d / c["table"], c["row"], c["column"]) for d in (GOLDEN / "v10_tables", REPORT / "tables")] == [c["v10"], c["new"]], c
    # at[i]: the position in report/'s tokens of v10's token i, where a block of the alignment keeps its length
    v10_tokens, now = V10_TOKENS.read_text().split(), report_text_numbers()
    at = {}
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, v10_tokens, now, autojunk=False).get_opcodes():
        if i2 - i1 == j2 - j1:
            at |= {i1 + k: j1 + k for k in range(i2 - i1)}
    for c in approved_changes("text"):
        assert (v10_tokens[c["token"]], now[at[c["token"]]]) == (c["v10"], c["new"]), c


def test_who_applies_the_approved_changes_is_described_alike():
    """This module's docstring, tests/golden/README.md, the about field of approved_number_changes.json and the CHANGELOG.md
    entry that introduced the file all name both tests that apply it: the report/ tests here and the real downstream test."""
    texts = {"docstring": __doc__, "README": (GOLDEN / "README.md").read_text(), "about": json.loads(APPROVED.read_text())["about"],
             "CHANGELOG": next(line for line in (REPO / "CHANGELOG.md").read_text().splitlines() if "approved_number_changes.json` lists them" in line)}
    assert [name for name, text in texts.items() if "real downstream test" not in " ".join(text.split())] == []
