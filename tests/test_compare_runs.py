"""Tests for study/compare_runs.py's comparison logic, on small synthetic trees built in tmp_path.
No real study output is needed here; the mini end-to-end run against real data is a later task."""
import json
import math
import sys
from pathlib import Path

import polars as pl
import pysam
import pytest

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


def test_json_int_vs_float_same_value_is_a_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "summary.json").write_text(json.dumps({"n": 1}))
    (b / "summary.json").write_text(json.dumps({"n": 1.0}))

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "n" in diffs[0]


def test_json_bool_vs_int_same_value_is_a_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "summary.json").write_text(json.dumps({"flag": True}))
    (b / "summary.json").write_text(json.dumps({"flag": 1}))

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "flag" in diffs[0]


def test_json_key_on_one_side_only_prints_its_path(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "summary.json").write_text(json.dumps({"q1": {"n": 3, "extra": 1}}))
    (b / "summary.json").write_text(json.dumps({"q1": {"n": 3}}))

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "q1.extra" in diffs[0]
    assert "missing on B" in diffs[0]


def test_json_ignored_keys_are_not_compared(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "report.json").write_text(json.dumps({
        "uuid": "11111111-1111-1111-1111-111111111111",
        "createdAt": "2026-10-05T12:00:00Z",
        "_comment": "Created by pbcopper v2.7.0",
        "value": 42,
    }))
    (b / "report.json").write_text(json.dumps({
        "uuid": "22222222-2222-2222-2222-222222222222",
        "createdAt": "2026-10-05T13:30:00Z",
        "_comment": "Created by pbcopper v2.8.0",
        "value": 42,
    }))

    assert compare_trees(a, b) == []


def test_json_ignored_key_missing_on_one_side_is_not_reported(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "report.json").write_text(json.dumps({"uuid": "11111111-1111-1111-1111-111111111111", "value": 1}))
    (b / "report.json").write_text(json.dumps({"value": 1}))

    assert compare_trees(a, b) == []


def _datastore_manifest(created_at: str, updated_at: str, file_created_at: str, file_modified_at: str,
                         unique_id: str, file_size: int, path: str, name: str) -> dict:
    return {
        "createdAt": created_at,
        "files": [
            {
                "createdAt": file_created_at,
                "description": "Split BAM file by lima",
                "fileSize": file_size,
                "fileTypeId": "PacBio.DataSet.ConsensusReadSet",
                "isChunked": False,
                "modifiedAt": file_modified_at,
                "name": name,
                "path": path,
                "sourceId": "barcoding.tasks.lima-0",
                "uniqueId": unique_id,
            }
        ],
        "updatedAt": updated_at,
        "version": "0.2.2",
    }


def test_json_datastore_manifest_ignored_keys_are_not_compared(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "lima").mkdir()
    (b / "lima").mkdir()
    manifest_a = _datastore_manifest(
        "2026-10-05T12:05:52.56Z", "2026-10-05T12:05:52.56Z", "2026-10-05T12:05:52.56Z",
        "2026-10-05T12:05:52.56Z", "f0903c78-2deb-4d2f-bb2d-3dedd9c01d55", -968319015,
        "sample.isoseq.FlashBD_5p--FlashBD_3p.consensusreadset.xml", "Barcodes FlashBD_5p--FlashBD_3p")
    manifest_b = _datastore_manifest(
        "2026-10-06T09:11:03.12Z", "2026-10-06T09:11:03.12Z", "2026-10-06T09:11:03.12Z",
        "2026-10-06T09:11:03.12Z", "a1b2c3d4-0000-0000-0000-000000000000", 123456,
        "sample.isoseq.FlashBD_5p--FlashBD_3p.consensusreadset.xml", "Barcodes FlashBD_5p--FlashBD_3p")
    (a / "lima" / "sample.isoseq.json").write_text(json.dumps(manifest_a))
    (b / "lima" / "sample.isoseq.json").write_text(json.dumps(manifest_b))

    assert compare_trees(a, b) == []


def test_json_datastore_manifest_path_difference_is_still_reported(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "lima").mkdir()
    (b / "lima").mkdir()
    manifest_a = _datastore_manifest(
        "2026-10-05T12:05:52.56Z", "2026-10-05T12:05:52.56Z", "2026-10-05T12:05:52.56Z",
        "2026-10-05T12:05:52.56Z", "f0903c78-2deb-4d2f-bb2d-3dedd9c01d55", -968319015,
        "sample.isoseq.FlashBD_5p--FlashBD_3p.consensusreadset.xml", "Barcodes FlashBD_5p--FlashBD_3p")
    manifest_b = _datastore_manifest(
        "2026-10-06T09:11:03.12Z", "2026-10-06T09:11:03.12Z", "2026-10-06T09:11:03.12Z",
        "2026-10-06T09:11:03.12Z", "a1b2c3d4-0000-0000-0000-000000000000", 123456,
        "sample.isoseq.OTHER_5p--OTHER_3p.consensusreadset.xml", "Barcodes FlashBD_5p--FlashBD_3p")
    (a / "lima" / "sample.isoseq.json").write_text(json.dumps(manifest_a))
    (b / "lima" / "sample.isoseq.json").write_text(json.dumps(manifest_b))

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "files[0].path" in diffs[0]


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


def test_tsv_crlf_line_endings_is_a_difference(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "table.tsv").write_text("zm\tgate\n1\tpass\n2\tfail\n")
    (b / "table.tsv").write_bytes(b"zm\tgate\r\n1\tpass\r\n2\tfail\r\n")

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "table.tsv" in diffs[0]
    assert "rows in a different order" not in capsys.readouterr().out


def test_tsv_missing_trailing_newline_is_a_difference(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "table.tsv").write_text("zm\tgate\n1\tpass\n")
    (b / "table.tsv").write_text("zm\tgate\n1\tpass")

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "table.tsv" in diffs[0]
    assert "rows in a different order" not in capsys.readouterr().out


def test_skipped_marker_file_ignored_even_if_it_differs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "logs").mkdir(parents=True)
    (b / "logs").mkdir(parents=True)
    (a / "logs" / "s01.ok").write_text("2026-01-01T00:00:00\n")
    (b / "logs" / "s01.ok").write_text("2026-11-03T12:00:00\n")

    assert compare_trees(a, b) == []


def test_xml_and_pbi_files_skipped_even_if_they_differ(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "lima").mkdir(parents=True)
    (b / "lima").mkdir(parents=True)
    (a / "lima" / "sample.consensusreadset.xml").write_text('<x UniqueId="aaa" CreatedAt="t1"/>')
    (b / "lima" / "sample.consensusreadset.xml").write_text('<x UniqueId="bbb" CreatedAt="t2"/>')
    (a / "lima" / "sample.bam.pbi").write_bytes(b"\x00\x01")
    (b / "lima" / "sample.bam.pbi").write_bytes(b"\x02\x03\x04")

    assert compare_trees(a, b) == []


def test_parquet_rows_reordered_with_zm_di_key_compare_equal(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _parquet(a / "segments.parquet", [1, 2, 3], [0, 0, 0], [1.5, 2.5, 3.5])
    _parquet(b / "segments.parquet", [3, 1, 2], [0, 0, 0], [3.5, 1.5, 2.5])

    assert compare_trees(a, b) == []


def test_parquet_duplicate_keys_in_different_row_order_compare_equal(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    # zm=1, di=0 appears twice on each side; the two files list those two rows in opposite
    # order. Sorting by the key columns alone is not guaranteed stable for the tie, so without
    # also sorting by "value" the two sides could end up misaligned and report a false diff.
    _parquet(a / "segments.parquet", [1, 1, 2], [0, 0, 0], [1.0, 2.0, 3.0])
    _parquet(b / "segments.parquet", [1, 1, 2], [0, 0, 0], [2.0, 1.0, 3.0])

    assert compare_trees(a, b) == []


def test_parquet_columns_differ(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _parquet(a / "segments.parquet", [1, 2], [0, 0], [1.5, 2.5])
    a_df = pl.read_parquet(a / "segments.parquet")
    b.mkdir()
    a_df.rename({"value": "other"}).write_parquet(b / "segments.parquet")

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "columns differ" in diffs[0]


def test_parquet_dtype_difference_is_a_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pl.DataFrame({"zm": [1, 2], "di": [0, 0], "value": [1, 2]},
                 schema={"zm": pl.UInt32, "di": pl.Int16, "value": pl.Int32}).write_parquet(a / "segments.parquet")
    pl.DataFrame({"zm": [1, 2], "di": [0, 0], "value": [1, 2]},
                 schema={"zm": pl.UInt32, "di": pl.Int16, "value": pl.Int64}).write_parquet(b / "segments.parquet")

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "dtypes differ" in diffs[0]


def test_parquet_null_vs_value_is_a_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pl.DataFrame({"zm": [1, 2], "di": [0, 0], "value": [None, 2.5]},
                 schema={"zm": pl.UInt32, "di": pl.Int16, "value": pl.Float64}).write_parquet(a / "segments.parquet")
    pl.DataFrame({"zm": [1, 2], "di": [0, 0], "value": [1.5, 2.5]},
                 schema={"zm": pl.UInt32, "di": pl.Int16, "value": pl.Float64}).write_parquet(b / "segments.parquet")

    diffs = compare_trees(a, b)
    assert len(diffs) == 1
    assert "zm=1" in diffs[0]


def test_parquet_nan_equals_nan_no_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pl.DataFrame({"zm": [1, 2], "di": [0, 0], "value": [math.nan, 2.5]},
                 schema={"zm": pl.UInt32, "di": pl.Int16, "value": pl.Float64}).write_parquet(a / "segments.parquet")
    pl.DataFrame({"zm": [1, 2], "di": [0, 0], "value": [math.nan, 2.5]},
                 schema={"zm": pl.UInt32, "di": pl.Int16, "value": pl.Float64}).write_parquet(b / "segments.parquet")

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
    assert any("record count differs" in d for d in diffs)
    assert any("record names differ" in d for d in diffs)


def test_missing_roots_exit_2(tmp_path):
    a, b = tmp_path / "nonexistent_a", tmp_path / "nonexistent_b"

    with pytest.raises(SystemExit) as exc_info:
        main([str(a), str(b)])
    assert exc_info.value.code == 2


def test_root_that_is_a_file_exits_2(tmp_path):
    a = tmp_path / "a"
    a.write_text("not a directory")
    b = tmp_path / "b"
    b.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        main([str(a), str(b)])
    assert exc_info.value.code == 2


def test_max_rows_zero_exits_2(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        main([str(a), str(b), "--max-rows", "0"])
    assert exc_info.value.code == 2


def test_max_rows_negative_exits_2(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        main([str(a), str(b), "--max-rows", "-1"])
    assert exc_info.value.code == 2


def test_main_summary_line_counts_differences_and_files(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "only_in_a.json").write_text("{}")
    (b / "only_in_b.json").write_text("{}")
    (a / "summary.json").write_text(json.dumps({"n": 1, "m": 2}))
    (b / "summary.json").write_text(json.dumps({"n": 9, "m": 8}))

    assert main([str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "4 differences in 3 files" in out


def test_main_no_differences_summary_line(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "summary.json").write_text('{"n": 1}')
    (b / "summary.json").write_text('{"n": 1}')

    assert main([str(a), str(b)]) == 0
    assert capsys.readouterr().out.strip() == "no differences"
