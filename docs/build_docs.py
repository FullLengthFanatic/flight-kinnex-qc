#!/usr/bin/env python3
"""Build README.md and docs/*.md from their sources README.md.in and docs/*.md.in. Every number comes from numbers.json,
printed by the Supplementary Note's own formatter (study/report/render.py): {key|format} prints a value, {lo..hi|format} a
range, with the formats the Note uses, so the docs and the Note print each value alike. Two slots take their numbers from
other outputs of the same run:
  <!--table false_cuts_by_library-->  skera's false cuts per library, from tables/fig1_skera_false_cut_anatomy.tsv
                                       (numbers.json holds most of these values only as the range over the libraries)
  <!--excerpt skera_false_cut-->       ten lines of examples/example_reads.fasta, record skera_false_cut
The claims the Note pins to the data (render.check_claims) are checked first; build stops on one that no longer holds.
The docs are built from the numbers.json and tables computed from the v10 inputs, tests/golden/v10_numbers.json and
tests/golden/v10_tables/ (the defaults), until the full rerun's report/numbers.json and report/tables/ are adopted; then
give --numbers report/numbers.json --tables report/tables.
Each generated file starts with a comment naming its source; tests/test_docs.py checks that they are up to date.
Usage: docs/build_docs.py [--numbers PATH] [--tables DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "study" / "report"))
from render import FORMATS, check_claims, fill  # noqa: E402

NUMBERS = REPO / "tests" / "golden" / "v10_numbers.json"
TABLES = REPO / "tests" / "golden" / "v10_tables"
FASTA = REPO / "examples" / "example_reads.fasta"
LIBRARIES = ("ND", "HM", "kin")
# The rows of the false-cut table: column of fig1_skera_false_cut_anatomy.tsv, label, format. "Those" are the false cuts
# whose skipped adapter is found, the base of false_cut_anatomy.py's missed_is_D, missed_same_letter_as_skera_call and fates.
FALSE_CUT_ROWS = [
    ("complete_passing_reads", "Complete arrays skera passes", "fmt"),
    ("false_cuts", "False cuts in them", "fmt"),
    ("reads_with_false_cut", "Complete arrays skera passes with at least one false cut", "pct1"),
    ("skera_calls_D", "False cuts that skera labels D", "pct0"),
    ("missed_adapter_found", "False cuts where we find the real adapter skera skipped", "pct1"),
    ("missed_is_D", "Of those, the skipped adapter is D", "pct0"),
    ("missed_same_letter_as_skera_call", "Of those, skera labels the cut with the skipped adapter's letter", "pct0"),
    ("fate_fused_lima", "Of those, lima passes the chimera", "pct1"),
    ("fate_frag_lima", "Of those, lima passes the fragment", "pct1"),
    ("median_cut_to_missed_adapter_bp", "Median distance from the skipped adapter to the cut (bp)", "f0"),
    ("proper_sread_lima_pass", "lima passes the proper S-reads of these arrays, for comparison", "pct1"),
]
# What the excerpt marks: the name a feature has in the FASTA header, and what it is.
EXCERPT_MARKS = {"C": "adapter C", "D(real)": "the real adapter D, which skera skips", "skera_D": "the look-alike where skera cuts and writes D"}


def false_cut_table(tables: Path) -> str:
    t = pl.read_csv(tables / "fig1_skera_false_cut_anatomy.tsv", separator="\t")
    rows = {r["library"]: r for r in t.iter_rows(named=True)}
    out = ["| | " + " | ".join(LIBRARIES) + " |", "| --- |" + " --- |" * len(LIBRARIES)]
    out += [f"| {label} | " + " | ".join(FORMATS[f](rows[lib][col]) for lib in LIBRARIES) + " |" for col, label, f in FALSE_CUT_ROWS]
    return "\n".join(out)


def excerpt(fasta: Path, name: str = "skera_false_cut") -> str:
    """Ten lines of the record: its header up to the first " | ", and the FASTA lines that hold the first adapter, the real D
    and the site skera called D, each line led by the position of its first base and followed by a line of carets under
    the feature, with the feature's name and positions as the header gives them (1-based, on the shown sequence)."""
    record = fasta.read_text().split(f">{name} ")[1].split("\n>")[0].splitlines()
    header, lines = f">{name} " + record[0], record[1:]
    wrap = len(lines[0])
    features = {}
    for item in header.rsplit(" | ", 1)[1].split(", "):
        label, span = item.rsplit(" ", 1)
        features.setdefault(label, tuple(int(x) for x in span.split("-")))
    width = len(str(len(lines) * wrap))
    out, last = [header.split(" | ")[0]], -1
    for label, what in EXCERPT_MARKS.items():
        start, end = features[label]
        first, final = (start - 1) // wrap, (end - 1) // wrap
        if first > last + 1:
            out.append("...")
        out += [f"{i * wrap + 1:>{width}} {lines[i]}" for i in range(first, final + 1)]
        col = (start - 1) % wrap if first == final else 0
        carets = "^" * (min(end, (final + 1) * wrap) - max(start, final * wrap + 1) + 1)
        above = " (it starts on the line above)" if first != final else ""
        out.append(" " * (width + 1 + col) + f"{carets} {label} {start}-{end}: {what}{above}")
        last = final
    return "\n".join(out)


def build(numbers_path: Path, tables: Path) -> dict[Path, str]:
    """The generated files, path -> text, from the .md.in sources with numbers.json at numbers_path and the tables in tables."""
    numbers = json.loads(numbers_path.read_text())
    check_claims(tables, numbers)
    slots = {"<!--table false_cuts_by_library-->": false_cut_table(tables),
             "<!--excerpt skera_false_cut-->": f"```text\n{excerpt(FASTA)}\n```"}
    out = {}
    for source in [REPO / "README.md.in", *sorted((REPO / "docs").glob("*.md.in"))]:
        text = source.read_text()
        for slot, value in slots.items():
            text = text.replace(slot, value)
        rel = source.relative_to(REPO).as_posix()
        head = f"<!-- Generated by docs/build_docs.py from {rel}; edit that file, then run docs/build_docs.py. -->\n"
        out[source.with_suffix("")] = head + fill(text, numbers)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--numbers", type=Path, default=NUMBERS, help="numbers.json (default: the v10 numbers in tests/golden)")
    ap.add_argument("--tables", type=Path, default=TABLES, help="the report's tables/ folder (default: the v10 tables in tests/golden)")
    args = ap.parse_args(argv)
    for path, text in build(args.numbers, args.tables).items():
        path.write_text(text)
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
