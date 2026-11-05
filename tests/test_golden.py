"""Golden regression of the Supplementary Note: rendered from the frozen v10 source tables (tests/golden/v10_tables) and the
numbers.json compute writes from the v10 inputs (tests/golden/v10_numbers.json), the numbers a reader sees are v10's
(tests/golden/v10_text_numbers.txt) apart from the changes listed in tests/golden/allowed_changes.txt, and the figures
and tables are first cited in order. Fig. 3 draws per-read data, which is not frozen: the test draws two made-up reads
(its plot is not visible text) and checks the caption's ZMWs, which come from numbers.json."""
import difflib
import json
import re

import pytest

from golden_text import numbers, visible_text
from pipeline_helpers import EXAMPLE_TRACKS, GOLDEN

import render

V10_TOKENS = GOLDEN / "v10_text_numbers.txt"
ALLOWED = GOLDEN / "allowed_changes.txt"


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
