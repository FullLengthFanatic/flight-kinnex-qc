"""Tests for study/report/figures.py: Supplementary Figs. 2 to 9 build from the report's tables and numbers.json, with the
traces, annotations, shapes and panels v10's figures had."""
import json

import pytest

from pipeline_helpers import GOLDEN

import figures
from kinnex_qc.scan import Hit

# (traces, annotations, shapes, x axes) of each figure as v10's build script made it from the v10 inputs.
V10_COUNTS = {"fig2": (7, 5, 0, 2), "fig4": (5, 3, 0, 2), "fig5": (7, 7, 0, 3), "fig6": (4, 3, 0, 1), "fig7": (9, 3, 0, 3),
              "fig8": (15, 6, 0, 6), "fig9": (7, 5, 1, 4)}


def counts(fig) -> tuple[int, int, int, int]:
    layout = fig.layout.to_plotly_json()
    return len(fig.data), len(fig.layout.annotations), len(fig.layout.shapes), sum(1 for k in layout if k.startswith("xaxis"))


@pytest.fixture(scope="module")
def data() -> dict:
    return figures.load(GOLDEN / "v10_tables", json.loads((GOLDEN / "v10_numbers.json").read_text()))


@pytest.mark.parametrize("name", sorted(V10_COUNTS))
def test_figure_builds_from_the_golden_tables_as_v10(data, name):
    assert counts(getattr(figures, name)(data)) == V10_COUNTS[name]


def test_panel_letters_left_aligned_on_their_panels(data):
    fig = figures.fig5(data)
    titles = [a for a in fig.layout.annotations if str(a.text).startswith("<b>")]
    assert [a.text for a in titles] == figures.panel_titles("a  Molecules per HiFi read", "b  skera on ND reads, by chain", "c  ND against kin")
    assert [a.x for a in titles] == [fig.layout[f"xaxis{'' if i == 0 else i + 1}"].domain[0] for i in range(3)]
    assert {a.xanchor for a in titles} == {"left"}


def test_fig3_draws_the_example_reads_from_their_track_data():
    reads = [{"zm": 5, "read_len": 1600, "chain": "B C", "pattern": "truncated",
              "features": [{"name": "B", "kind": "MAS", "strand": "+", "start": 0, "end": 17, "edits": 0},
                           {"name": "C", "kind": "MAS", "strand": "+", "start": 1500, "end": 1517, "edits": 1}],
              "segments": [{"start": 17, "end": 1500, "end_class": "proper", "polyT_len": None, "tso_side": None}],
              "skera": [[17, 1500]]},
             {"zm": 2, "read_len": 900, "chain": "", "pattern": "unresolved", "features": [], "segments": [], "skera": []}]
    fig = figures.fig3(reads)
    assert list(fig.layout.yaxis.ticktext) == ["5", "2"]                 # in the order given
    assert (fig.layout.width, fig.layout.height) == (1040, 760) and fig.layout.title.text == ""
    assert [t for t in fig.data if t.name == "C" and t.type == "bar"][0].hovertext[0] == "C + 1,500-1,517, 1 edits"
    assert figures.track_hits(reads[0])["features"][1] == Hit("C", "MAS", "+", 1500, 1517, 1)
