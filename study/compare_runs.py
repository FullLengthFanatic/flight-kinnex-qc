"""Exact comparison of two study output trees.

`compare_runs.py A B` walks both trees and compares files by relative path: missing on either
side is a difference. Parquet frames are read with polars, sorted by their key columns, and
compared value by value (nulls equal, NaN equal to NaN, no tolerance). TSV tables are compared as
text, but rows in a different order are a note, not a difference. JSON is compared recursively by
key path, ignoring a fixed set of PacBio run-identity keys (see JSON_IGNORED_KEYS). BAMs are
compared only with --bam-names (record count and the set of record names); otherwise they are
listed as not compared. Everything else is compared byte for byte.

Run bookkeeping that differs by run time or isn't reproducible byte for byte is skipped
everywhere: logs/, *.log, *.ok and *.done.json (run-time markers and records), anything under a
figures/ directory (rendered plots), .staging/ (in-progress files), and (controller ruling
2026-11-03) *.xml and *.pbi (PacBio tool metadata that records run identity, not analysis output).

Prints one line per difference, `relative/path: what differs`, then a last line `N differences in
M files` (or `no differences`). Exit 1 if any difference, 0 otherwise, 2 on bad arguments.
"""
import argparse
import filecmp
import json
import math
import sys
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import polars as pl
import pysam

# Run bookkeeping skipped everywhere, never compared even if it differs. A trailing "/" marks a
# directory component anywhere in the path (figures/ holds rendered plots, not reproducible byte
# for byte; .staging/ holds in-progress files); otherwise the entry is a filename suffix (*.log
# and *.ok are run logs and per-step completion markers; *.done.json records run times; *.xml is
# PacBio dataset metadata whose UniqueId/CreatedAt/Name/TimeStampedName attributes record run
# identity, not analysis output; *.pbi is a BAM index whose byte offsets depend on the @PG lines
# lima/isoseq stamp with the run's own output paths, so it never matches byte for byte across
# runs, per controller ruling 2026-11-03).
SKIP = ("logs/", ".log", ".ok", ".done.json", "figures/", ".staging/", ".xml", ".pbi")

# JSON keys ignored wherever they occur, at any depth, on either side: PacBio tool run identity,
# not analysis output (controller ruling 2026-11-03). A key present on only one side because of
# this is not reported as missing either.
JSON_IGNORED_KEYS = {
    "uuid",       # random id lima/isoseq assigns to a dataset or report, regenerated every run
    "createdAt",  # PacBio tool timestamp on a dataset, file or report entry
    "_comment",   # pbcommand's report-generation timestamp comment (e.g. "Created by pbcopper v2.7.0")
}

KEY_COLUMN_SETS = (("zm", "di"), ("zm", "seg_index"), ("zm", "start"))


class Diff(NamedTuple):
    """One comparison result for a single file: `kind` is "difference" (counted, fails the run)
    or "note" (informational only, e.g. a TSV's rows being reordered or a BAM left uncompared)."""
    path: str
    kind: str
    text: str


def _is_skipped(rel: PurePosixPath) -> bool:
    for pattern in SKIP:
        if pattern.endswith("/"):
            if pattern[:-1] in rel.parts[:-1]:
                return True
        elif rel.name.endswith(pattern):
            return True
    return False


def _collect_files(root: Path) -> set[PurePosixPath]:
    files = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = PurePosixPath(p.relative_to(root).as_posix())
        if not _is_skipped(rel):
            files.add(rel)
    return files


def _key_columns(columns: list[str]) -> list[str]:
    for keys in KEY_COLUMN_SETS:
        if all(k in columns for k in keys):
            return list(keys)
    return columns


def _compare_parquet(rel: PurePosixPath, path_a: Path, path_b: Path, max_rows: int) -> list[Diff]:
    df_a = pl.read_parquet(path_a)
    df_b = pl.read_parquet(path_b)
    if df_a.columns != df_b.columns:
        return [Diff(str(rel), "difference", f"columns differ: {df_a.columns} vs {df_b.columns}")]
    if df_a.dtypes != df_b.dtypes:
        return [Diff(str(rel), "difference", f"dtypes differ: {df_a.dtypes} vs {df_b.dtypes}")]

    # Sort by the key columns, then by every other column, on both sides: polars' sort is not
    # guaranteed stable, so with duplicate keys, sorting by the keys alone could leave the two
    # sides' tied rows in different relative order and report a false difference. Sorting by the
    # full column tuple gives both sides the same well-defined order regardless.
    keys = _key_columns(df_a.columns)
    sort_cols = keys + [c for c in df_a.columns if c not in keys]
    df_a = df_a.sort(sort_cols)
    df_b = df_b.sort(sort_cols)
    if df_a.height != df_b.height:
        return [Diff(str(rel), "difference", f"row count differs: {df_a.height} vs {df_b.height}")]
    if df_a.height == 0:
        return []

    # Per-column equality, nulls equal and (polars default) NaN equal to NaN; no tolerance.
    col_eq = {}
    for c in df_a.columns:
        a_col, b_col = df_a[c], df_b[c]
        col_eq[c] = (a_col == b_col).fill_null(False) | (a_col.is_null() & b_col.is_null())

    mismatch = pl.repeat(False, df_a.height, dtype=pl.Boolean, eager=True)
    for eq in col_eq.values():
        mismatch = mismatch | ~eq
    if not mismatch.any():
        return []

    diffs = []
    for i in mismatch.arg_true().head(max_rows).to_list():
        key_text = ", ".join(f"{k}={df_a[k][i]!r}" for k in keys)
        changed = [f"{c}: {df_a[c][i]!r} != {df_b[c][i]!r}" for c in df_a.columns if not col_eq[c][i]]
        diffs.append(Diff(str(rel), "difference", f"row [{key_text}] differs ({', '.join(changed)})"))
    return diffs


def _compare_tsv(rel: PurePosixPath, path_a: Path, path_b: Path, max_rows: int) -> list[Diff]:
    # Decoded from bytes, not Path.read_text(): read_text() applies universal-newline
    # translation, which would silently turn CRLF into LF and hide a line-ending difference
    # before it's ever compared.
    text_a = path_a.read_bytes().decode()
    text_b = path_b.read_bytes().decode()
    if text_a == text_b:
        return []

    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()
    # splitlines() strips the line terminator, so it cannot tell CRLF from LF, or a missing
    # trailing newline from one that's present. Same lines in the same order with text_a !=
    # text_b therefore means the text differs only in that encoding, not in content or order:
    # a real difference, not the "rows in a different order" note below.
    if lines_a == lines_b:
        return [Diff(str(rel), "difference", "text differs (line endings or trailing newline)")]
    if lines_a[:1] == lines_b[:1] and sorted(lines_a[1:]) == sorted(lines_b[1:]):
        return [Diff(str(rel), "note", "rows in a different order")]

    diffs = []
    for i in range(max(len(lines_a), len(lines_b))):
        la = lines_a[i] if i < len(lines_a) else None
        lb = lines_b[i] if i < len(lines_b) else None
        if la != lb:
            diffs.append(Diff(str(rel), "difference", f"row {i}: {la!r} != {lb!r}"))
            if len(diffs) >= max_rows:
                break
    return diffs


def _json_values_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _json_diffs(path: str, a, b) -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = []
        for key in sorted(set(a) | set(b)):
            if key in JSON_IGNORED_KEYS:
                continue
            sub = f"{path}.{key}" if path else key
            if key not in a:
                diffs.append(f"{sub}: missing on A")
            elif key not in b:
                diffs.append(f"{sub}: missing on B")
            else:
                diffs.extend(_json_diffs(sub, a[key], b[key]))
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: list length differs: {len(a)} vs {len(b)}"]
        diffs = []
        for i, (av, bv) in enumerate(zip(a, b)):
            diffs.extend(_json_diffs(f"{path}[{i}]", av, bv))
        return diffs
    if _json_values_equal(a, b):
        return []
    return [f"{path}: {a!r} != {b!r}"]


def _compare_json(rel: PurePosixPath, path_a: Path, path_b: Path) -> list[Diff]:
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())
    return [Diff(str(rel), "difference", d) for d in _json_diffs("", a, b)]


def _bam_names(path: Path) -> tuple[set[str], int]:
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        names = set()
        n = 0
        for record in bam:
            names.add(record.query_name)
            n += 1
        return names, n


def _compare_bam(rel: PurePosixPath, path_a: Path, path_b: Path) -> list[Diff]:
    names_a, n_a = _bam_names(path_a)
    names_b, n_b = _bam_names(path_b)
    diffs = []
    if n_a != n_b:
        diffs.append(Diff(str(rel), "difference", f"record count differs: {n_a} vs {n_b}"))
    if names_a != names_b:
        diffs.append(Diff(str(rel), "difference",
                           f"record names differ: {len(names_a - names_b)} only in A, {len(names_b - names_a)} only in B"))
    return diffs


def _compare_bytes(rel: PurePosixPath, path_a: Path, path_b: Path) -> list[Diff]:
    if filecmp.cmp(path_a, path_b, shallow=False):
        return []
    return [Diff(str(rel), "difference", "bytes differ")]


def _compare_file(rel: PurePosixPath, path_a: Path, path_b: Path, bam_names: bool, max_rows: int) -> list[Diff]:
    suffix = rel.suffix
    if suffix == ".parquet":
        return _compare_parquet(rel, path_a, path_b, max_rows)
    if suffix == ".tsv":
        return _compare_tsv(rel, path_a, path_b, max_rows)
    if suffix == ".json":
        return _compare_json(rel, path_a, path_b)
    if suffix == ".bam":
        if not bam_names:
            return [Diff(str(rel), "note", "not compared")]
        return _compare_bam(rel, path_a, path_b)
    return _compare_bytes(rel, path_a, path_b)


def _compare_tree_results(a: Path, b: Path, bam_names: bool, max_rows: int) -> list[Diff]:
    files_a = _collect_files(a)
    files_b = _collect_files(b)

    results = []
    for rel in sorted(files_a - files_b):
        results.append(Diff(str(rel), "difference", f"missing in {b}"))
    for rel in sorted(files_b - files_a):
        results.append(Diff(str(rel), "difference", f"missing in {a}"))
    for rel in sorted(files_a & files_b):
        results.extend(_compare_file(rel, a / rel, b / rel, bam_names, max_rows))
    return results


def compare_trees(a: Path | str, b: Path | str, bam_names: bool = False, max_rows: int = 10) -> list[str]:
    """Compare the output trees rooted at a and b, file by file, skipping run bookkeeping (see
    SKIP). Returns the differences, one string each. A TSV whose rows are merely reordered, and a
    BAM compared with bam_names left False, print a note instead of adding to this list."""
    results = _compare_tree_results(Path(a), Path(b), bam_names, max_rows)
    for r in results:
        if r.kind == "note":
            print(f"{r.path}: {r.text}")
    return [f"{r.path}: {r.text}" for r in results if r.kind == "difference"]


def _max_rows(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"--max-rows must be >= 1, got {value}")
    return n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="compare_runs.py", description="Compare two study output trees file by file.")
    ap.add_argument("a", help="first output tree")
    ap.add_argument("b", help="second output tree")
    ap.add_argument("--bam-names", action="store_true",
                     help="also compare BAM record counts and names (default: list BAMs as not compared)")
    ap.add_argument("--max-rows", type=_max_rows, default=10,
                     help="differing rows printed per parquet or TSV file (default: 10)")
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not Path(args.a).is_dir():
        parser.error(f"not a directory: {args.a}")
    if not Path(args.b).is_dir():
        parser.error(f"not a directory: {args.b}")
    results = _compare_tree_results(Path(args.a), Path(args.b), args.bam_names, args.max_rows)
    for r in results:
        print(f"{r.path}: {r.text}")
    diffs = [r for r in results if r.kind == "difference"]
    if diffs:
        files = {r.path for r in diffs}
        print(f"{len(diffs)} differences in {len(files)} files")
    else:
        print("no differences")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
