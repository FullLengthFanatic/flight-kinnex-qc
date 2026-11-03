"""Exact comparison of two study output trees.

`compare_runs.py A B` walks both trees and compares files by relative path: missing on either
side is a difference. Parquet frames are read with polars, sorted by their key columns, and
compared value by value (nulls equal, NaN equal to NaN, no tolerance). TSV tables are compared as
text, but rows in a different order are a note, not a difference. JSON is compared recursively by
key path. BAMs are compared only with --bam-names (record count and the set of record names);
otherwise they are listed as not compared. Everything else is compared byte for byte.

Run bookkeeping that differs by run time or isn't reproducible byte for byte is skipped
everywhere: logs/, *.log, *.ok and *.done.json (run-time markers and records), anything under a
figures/ directory (rendered plots), and .staging/ (in-progress files).

Prints one line per difference, `relative/path: what differs`, then a last line `N differences in
M files` (or `no differences`). Exit 1 if any difference, 0 otherwise, 2 on bad arguments.
"""
import argparse
import json
import math
import sys
from pathlib import Path, PurePosixPath

import polars as pl
import pysam

# Run bookkeeping skipped everywhere, never compared even if it differs. A trailing "/" marks a
# directory component anywhere in the path (figures/ holds rendered plots, not reproducible byte
# for byte; .staging/ holds in-progress files); otherwise the entry is a filename suffix (*.log
# and *.ok are run logs and per-step completion markers; *.done.json records run times).
SKIP = ("logs/", ".log", ".ok", ".done.json", "figures/", ".staging/")

KEY_COLUMN_SETS = (("zm", "di"), ("zm", "seg_index"), ("zm", "start"))


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


def _compare_parquet(rel: PurePosixPath, path_a: Path, path_b: Path, max_rows: int) -> list[str]:
    df_a = pl.read_parquet(path_a)
    df_b = pl.read_parquet(path_b)
    if df_a.columns != df_b.columns:
        return [f"{rel}: columns differ: {df_a.columns} vs {df_b.columns}"]
    if df_a.dtypes != df_b.dtypes:
        return [f"{rel}: dtypes differ: {df_a.dtypes} vs {df_b.dtypes}"]

    keys = _key_columns(df_a.columns)
    df_a = df_a.sort(keys)
    df_b = df_b.sort(keys)
    if df_a.height != df_b.height:
        return [f"{rel}: row count differs: {df_a.height} vs {df_b.height}"]
    if df_a.height == 0:
        return []

    # Per-column equality, nulls equal and (polars default) NaN equal to NaN; no tolerance.
    col_eq = {}
    for c in df_a.columns:
        a_col, b_col = df_a[c], df_b[c]
        col_eq[c] = (a_col == b_col).fill_null(False) | (a_col.is_null() & b_col.is_null())

    mismatch = pl.Series([False] * df_a.height, dtype=pl.Boolean)
    for eq in col_eq.values():
        mismatch = mismatch | ~eq
    if not mismatch.any():
        return []

    diffs = []
    for i in mismatch.arg_true().head(max_rows).to_list():
        key_text = ", ".join(f"{k}={df_a[k][i]!r}" for k in keys)
        changed = [f"{c}: {df_a[c][i]!r} != {df_b[c][i]!r}" for c in df_a.columns if not col_eq[c][i]]
        diffs.append(f"{rel}: row [{key_text}] differs ({', '.join(changed)})")
    return diffs


def _compare_tsv(rel: PurePosixPath, path_a: Path, path_b: Path, max_rows: int) -> list[str]:
    text_a = path_a.read_text()
    text_b = path_b.read_text()
    if text_a == text_b:
        return []

    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()
    if lines_a[:1] == lines_b[:1] and sorted(lines_a[1:]) == sorted(lines_b[1:]):
        print(f"{rel}: rows in a different order")
        return []

    diffs = []
    for i in range(max(len(lines_a), len(lines_b))):
        la = lines_a[i] if i < len(lines_a) else None
        lb = lines_b[i] if i < len(lines_b) else None
        if la != lb:
            diffs.append(f"{rel}: row {i}: {la!r} != {lb!r}")
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


def _compare_json(rel: PurePosixPath, path_a: Path, path_b: Path) -> list[str]:
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())
    return [f"{rel}: {d}" for d in _json_diffs("", a, b)]


def _bam_names(path: Path) -> tuple[set[str], int]:
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        names = set()
        n = 0
        for record in bam:
            names.add(record.query_name)
            n += 1
        return names, n


def _compare_bam(rel: PurePosixPath, path_a: Path, path_b: Path) -> list[str]:
    names_a, n_a = _bam_names(path_a)
    names_b, n_b = _bam_names(path_b)
    diffs = []
    if n_a != n_b:
        diffs.append(f"{rel}: record count differs: {n_a} vs {n_b}")
    if names_a != names_b:
        diffs.append(f"{rel}: record names differ: {len(names_a - names_b)} only in A, {len(names_b - names_a)} only in B")
    return diffs


def _compare_bytes(rel: PurePosixPath, path_a: Path, path_b: Path) -> list[str]:
    if path_a.read_bytes() == path_b.read_bytes():
        return []
    return [f"{rel}: bytes differ"]


def _compare_file(rel: PurePosixPath, path_a: Path, path_b: Path, bam_names: bool, max_rows: int) -> list[str]:
    suffix = rel.suffix
    if suffix == ".parquet":
        return _compare_parquet(rel, path_a, path_b, max_rows)
    if suffix == ".tsv":
        return _compare_tsv(rel, path_a, path_b, max_rows)
    if suffix == ".json":
        return _compare_json(rel, path_a, path_b)
    if suffix == ".bam":
        if not bam_names:
            print(f"{rel}: not compared")
            return []
        return _compare_bam(rel, path_a, path_b)
    return _compare_bytes(rel, path_a, path_b)


def compare_trees(a: Path | str, b: Path | str, bam_names: bool = False, max_rows: int = 10) -> list[str]:
    """Compare the output trees rooted at a and b, file by file, skipping run bookkeeping (see
    SKIP). Returns the differences, one string each. A TSV whose rows are merely reordered, and a
    BAM compared with bam_names left False, print a note instead of adding to this list."""
    a, b = Path(a), Path(b)
    files_a = _collect_files(a)
    files_b = _collect_files(b)

    diffs = []
    for rel in sorted(files_a - files_b):
        diffs.append(f"{rel}: missing in {b}")
    for rel in sorted(files_b - files_a):
        diffs.append(f"{rel}: missing in {a}")
    for rel in sorted(files_a & files_b):
        diffs.extend(_compare_file(rel, a / rel, b / rel, bam_names, max_rows))
    return diffs


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
    diffs = compare_trees(args.a, args.b, bam_names=args.bam_names, max_rows=args.max_rows)
    for line in diffs:
        print(line)
    if diffs:
        files = {line.split(":", 1)[0] for line in diffs}
        print(f"{len(diffs)} differences in {len(files)} files")
    else:
        print("no differences")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
