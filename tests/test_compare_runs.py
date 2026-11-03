"""Tests for study/compare_runs.py's comparison logic, on small synthetic trees built in tmp_path.
No real study output is needed here; the mini end-to-end run against real data is a later task."""
import json
import math
import sys
from pathlib import Path

import polars as pl
import pysam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "study"))

from compare_runs import compare_trees, main  # noqa: E402


def _parquet(path: Path, zm: list[int], di: list[int], value: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"zm": zm, "di": di, "value": value},
                 schema={"zm": pl.UInt32, "di": pl.Int16, "value": pl.Float64}).write_parquet(path)


def test_identical_trees_no_differences(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _parquet(a / "sample" / "segments.parquet", [1, 2, 3], [0, 0, 0], [1.5, 2.5, 3.5])
    _parquet(b / "sample" / "segments.parquet", [1, 2, 3], [0, 0, 0], [1.5, 2.5, 3.5])
    (a / "sample" / "gates.tsv").write_text("zm\tgate\n1\tpass\n")
    (b / "sample" / "gates.tsv").write_text("zm\tgate\n1\tpass\n")
    (a / "sample" / "summary.json").write_text('{"n": 3}')
    (b / "sample" / "summary.json").write_text('{"n": 3}')

    assert main([str(a), str(b)]) == 0
    assert "no differences" in capsys.readouterr().out


def test_parquet_float_changed_in_last_bit_is_a_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    value = 1.5
    _parquet(a / "segments.parquet", [1, 2], [0, 0], [value, 2.5])
    _parquet(b / "segments.parquet", [1, 2], [0, 0], [math.nextafter(value, math.inf), 2.5])

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "segments.parquet" in diffs[0]


def test_json_float_changed_in_last_bit_is_a_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    value = 0.1
    (a / "summary.json").write_text(json.dumps({"q1": {"rate": value}}))
    (b / "summary.json").write_text(json.dumps({"q1": {"rate": math.nextafter(value, math.inf)}}))

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "q1.rate" in diffs[0]


def test_missing_and_extra_file(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "only_in_a.json").write_text("{}")
    (b / "only_in_b.json").write_text("{}")

    diffs = compare_trees(a, b)
    assert len(diffs) == 2
    joined = "\n".join(diffs)
    assert "only_in_a.json" in joined
    assert "only_in_b.json" in joined


def test_tsv_rows_reordered_gives_a_note_not_a_difference(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "table.tsv").write_text("zm\tgate\n1\tpass\n2\tfail\n")
    (b / "table.tsv").write_text("zm\tgate\n2\tfail\n1\tpass\n")

    diffs = compare_trees(a, b)
    assert diffs == []
    assert "rows in a different order" in capsys.readouterr().out


def test_skipped_marker_file_ignored_even_if_it_differs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "logs").mkdir(parents=True)
    (b / "logs").mkdir(parents=True)
    (a / "logs" / "s01.ok").write_text("2026-01-01T00:00:00\n")
    (b / "logs" / "s01.ok").write_text("2026-11-03T12:00:00\n")

    assert compare_trees(a, b) == []


def test_parquet_rows_reordered_with_zm_di_key_compare_equal(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _parquet(a / "segments.parquet", [1, 2, 3], [0, 0, 0], [1.5, 2.5, 3.5])
    _parquet(b / "segments.parquet", [3, 1, 2], [0, 0, 0], [3.5, 1.5, 2.5])

    assert compare_trees(a, b) == []


def _bam(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pysam.AlignmentFile(str(path), "wb", header={"HD": {"VN": "1.6"}}) as out:
        for name in names:
            a = pysam.AlignedSegment()
            a.query_name = name
            a.query_sequence = "ACGT"
            a.flag = 4
            out.write(a)


def test_bam_not_compared_without_bam_names_even_if_it_differs(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _bam(a / "molecules.bam", ["m/1/ccs/0"])
    _bam(b / "molecules.bam", ["m/2/ccs/0", "m/3/ccs/0"])

    assert compare_trees(a, b) == []
    assert "molecules.bam: not compared" in capsys.readouterr().out


def test_bam_names_compares_record_count_and_names(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _bam(a / "molecules.bam", ["m/1/ccs/0"])
    _bam(b / "molecules.bam", ["m/2/ccs/0", "m/3/ccs/0"])

    diffs = compare_trees(a, b, bam_names=True)
    assert len(diffs) == 2
    assert all("molecules.bam" in d for d in diffs)
