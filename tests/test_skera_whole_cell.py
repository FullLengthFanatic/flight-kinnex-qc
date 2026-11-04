"""Tests for study/analyses/skera_whole_cell.py: skera's own whole-cell reports copied into the output tree, byte for
byte, one <sample>.json per sample of the table."""
import pytest

from pipeline_helpers import GOLDEN_INPUTS, golden_diffs, run_analysis

HEADER = "sample\tcell\tdepletion\thifi_bam\tfail_bam\tsegmented_bam\tnon_passing_bam\tzmw_metrics\tskera_report\tfltnc_bam\n"
SAMPLES = ("BD144_ND", "BD144_HM", "BD144_kin")


def samples_table(path, reports: dict) -> None:
    path.write_text(HEADER + "".join(f"{s}\tcell\tnone" + "\tx" * 5 + f"\t{r}\tx\n" for s, r in reports.items()))


def test_copies_every_sample_report_byte_for_byte(tmp_path):
    reports = {}
    for s, text in (("S1", '{"attributes": [{"id": "read_segmentation.reads", "value": 7}]}\n'), ("S2", "{ }")):
        reports[s] = tmp_path / f"{s}_report.json"
        reports[s].write_text(text)
    samples_table(tmp_path / "samples.tsv", reports)
    run_analysis("skera_whole_cell.py", "--samples", str(tmp_path / "samples.tsv"), "--out", str(tmp_path / "out"))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["S1.json", "S2.json"]
    for s, path in reports.items():
        assert (tmp_path / "out" / f"{s}.json").read_bytes() == path.read_bytes()


@pytest.mark.real
def test_whole_cell_reports_equal_golden(local, tmp_path):
    samples_table(tmp_path / "samples.tsv", {s: local["inputs_dir"] / s / "read_segmentation.report.json" for s in SAMPLES})
    run_analysis("skera_whole_cell.py", "--samples", str(tmp_path / "samples.tsv"), "--out", str(tmp_path / "out"))
    golden = GOLDEN_INPUTS / "skera_whole_cell"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(p.name for p in golden.iterdir())
    assert golden_diffs(tmp_path / "out", golden, tmp_path / "cmp") == []
    for s in SAMPLES:
        assert (tmp_path / "out" / f"{s}.json").read_bytes() == (golden / f"{s}.json").read_bytes()
