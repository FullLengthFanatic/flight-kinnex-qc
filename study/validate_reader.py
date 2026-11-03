"""Validate `kinnex-qc run`'s summary.json against the frozen v10 study numbers, sample by sample.

Usage: study/validate_reader.py RUNS_DIR [--golden tests/golden]

RUNS_DIR/<sample>/summary.json for BD144_ND, BD144_HM and BD144_kin against the golden tables
and JSON files the study's own analyses wrote (tests/golden/v10_inputs, tests/golden/v10_tables).
Equality throughout is Python `==`, floats included: no tolerance, no rounding. Prints one line
per compared number (sample, key, the reader's value, v10's value, OK or DIFF) and exits 1 if any
line is a DIFF, 0 otherwise, except for the handful of keys in KNOWN_DIFFS: a DIFF there still
prints (nothing is hidden) but does not fail the run, and a one-line reason for it prints
separately at the end. See the task-13 controller decisions for exactly which summary.json key
compares against which golden file, and why; see CHANGELOG.md for each KNOWN_DIFFS entry's reason.
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

SAMPLES = ["BD144_ND", "BD144_HM", "BD144_kin"]
LIBRARY = {"BD144_ND": "ND", "BD144_HM": "HM", "BD144_kin": "kin"}

# Intended differences, each already documented and approved before this task (see CHANGELOG.md):
# a DIFF on one of these keys still prints, so nothing is hidden, but is not counted against the
# exit code. (sample, key) -> reason.
KNOWN_DIFFS = {
    ("BD144_ND", "junction_composition[TSO-TSO only].skera_pass_share"):
        "the reader derives skera_passing from S-read labels alone, with no --non-passing BAM "
        "(kinnex_qc.skera.passing_from_labels); it disagrees with skera's real verdict on 20 of "
        "ND's 1,969,703 hash-sampled reads, one of which falls in this bucket and so shifts its "
        "share by 1 read out of 313,852",
}

ARTIFACT_KEYS = ["proper_per_1000", "tso_tso_per_1000", "r1_r1_per_1000", "tso_only_per_1000",
                 "r1_only_per_1000", "none_per_1000", "other_per_1000", "internal_per_1000"]
# budget keys the reader and the golden fig6 table name the same; usable_not_clean (checked
# separately below) is the one budget number the golden table names not_written_cleanly instead.
BUDGET_KEYS = ["molecules", "unfilled_slots", "artifact_tso_tso", "artifact_r1_r1",
               "artifact_one_primer_or_none", "artifact_fused", "usable"]
MIRROR_ARM_CLASSES = ["CLEAN", "FUSED", "PARTIAL", "NOT_WRITTEN"]


def _check(lines: list[str], ok: list[bool], notes: list[str], sample: str, key: str, reader, v10) -> None:
    """Append one printed line (sample, key, reader's value, v10's value, OK/DIFF) and record
    whether it matched. `==` is exact for floats too: no rounding, no tolerance.

    A DIFF on a KNOWN_DIFFS key still prints as DIFF, but is not counted against the exit code;
    a one-line reason goes to `notes` instead, printed separately after every comparison."""
    good = reader == v10
    lines.append(f"{sample}\t{key}\t{reader!r}\t{v10!r}\t{'OK' if good else 'DIFF'}")
    if good:
        ok.append(True)
        return
    reason = KNOWN_DIFFS.get((sample, key))
    if reason is None:
        ok.append(False)
    else:
        ok.append(True)
        notes.append(f"{sample}\t{key}: documented difference, not counted as a failure ({reason})")


def _rows_by_key(df: pl.DataFrame, key_cols: list[str], none_as: str = "none") -> dict:
    """df's rows as a dict keyed by key_cols (a single value if key_cols has one column, else a
    tuple), a null in a key column standing for `none_as` (the false-cut class table uses an empty
    TSV cell for "no adapter letter")."""
    rows = {}
    for row in df.to_dicts():
        key = tuple(none_as if row[c] is None else row[c] for c in key_cols)
        rows[key[0] if len(key_cols) == 1 else key] = row
    return rows


def _compare_top_level(lines: list[str], ok: list[bool], notes: list[str], sample: str, summary: dict, golden: Path) -> None:
    q2 = json.loads((golden / "v10_inputs" / "hash_0.18" / sample / "q2" / "q2_summary.json").read_text())
    _check(lines, ok, notes, sample, "hifi_reads", summary["hifi_reads"], q2["hifi_reads"])
    _check(lines, ok, notes, sample, "hifi_reads_in_file", summary["hifi_reads_in_file"], q2["hifi_reads_total"])
    for k in ("complete_fraction", "mirror_fraction"):
        _check(lines, ok, notes, sample, f"fractions.{k}", summary["fractions"][k], q2[k])
    for k in ARTIFACT_KEYS:
        _check(lines, ok, notes, sample, f"artifacts.{k}", summary["artifacts"][k], q2[k])
    for k in ("mirror_junctions", "mirror_junction_tso_tso_share", "mirror_junction_r1_r1_share"):
        _check(lines, ok, notes, sample, f"mirror_junction_shares.{k}", summary["mirror_junction_shares"][k], q2[k])


def _compare_patterns(lines: list[str], ok: list[bool], notes: list[str], sample: str, summary: dict, golden: Path) -> None:
    gold = pl.read_csv(golden / "v10_tables" / "fig2a_chain_patterns.tsv", separator="\t").filter(pl.col("library") == LIBRARY[sample])
    gold_by_pattern = _rows_by_key(gold, ["pattern"])
    for row in summary["patterns"]:
        gold_row = gold_by_pattern[row["pattern"]]
        for k in ("reads", "share"):
            _check(lines, ok, notes, sample, f"patterns[{row['pattern']}].{k}", row[k], gold_row[k])


def _compare_skera(lines: list[str], ok: list[bool], notes: list[str], sample: str, summary: dict, golden: Path) -> None:
    sk = summary["skera"]
    fc_gold = json.loads((golden / "v10_inputs" / "false_cut_anatomy" / f"{sample}_false_cuts.json").read_text())
    _check(lines, ok, notes, sample, "skera.complete_passing_reads", sk["complete_passing_reads"], fc_gold["complete_passing_reads"])
    _check(lines, ok, notes, sample, "skera.false_cuts", sk["false_cuts"], fc_gold["false_cuts"])
    _check(lines, ok, notes, sample, "skera.false_cut_reads_share", sk["false_cut_reads_share"], fc_gold["reads_with_false_cut"])
    _check(lines, ok, notes, sample, "skera.false_cuts_per_read", sk["false_cuts_per_read"], fc_gold["false_cuts_per_read"])

    classes = pl.read_csv(golden / "v10_inputs" / "false_cut_anatomy" / f"{sample}_false_cut_classes.tsv", separator="\t")
    letter_tables = {"false_cut_letters": _rows_by_key(classes.filter(pl.col("field") == "skera_letter"), ["value"]),
                      "skipped_adapter_letters": _rows_by_key(classes.filter(pl.col("field") == "missed"), ["value"])}
    for label, gold_rows in letter_tables.items():
        reader_dict = sk[label]
        for letter in sorted(set(reader_dict) | set(gold_rows)):
            got = reader_dict.get(letter)
            want = gold_rows[letter]["false_cuts"] if letter in gold_rows else None
            _check(lines, ok, notes, sample, f"skera.{label}[{letter}]", got, want)

    budget_gold = pl.read_csv(golden / "v10_tables" / "fig6_molecule_budget.tsv", separator="\t").filter(pl.col("library") == LIBRARY[sample]).to_dicts()[0]
    for k in BUDGET_KEYS:
        _check(lines, ok, notes, sample, f"skera.budget.{k}", sk["budget"][k], budget_gold[k])
    _check(lines, ok, notes, sample, "skera.budget.usable_not_clean", sk["budget"]["usable_not_clean"], budget_gold["not_written_cleanly"])


def _compare_nd_only(lines: list[str], ok: list[bool], notes: list[str], sample: str, summary: dict, golden: Path) -> None:
    junctions_gold = _rows_by_key(pl.read_csv(golden / "v10_tables" / "fig4b_mirror_junction_letters_ND.tsv", separator="\t"),
                                   ["letter", "molecule"])
    for row in summary["junctions"]:
        gold_row = junctions_gold[(row["letter"], row["molecule"])]
        _check(lines, ok, notes, sample, f"junctions[{row['letter']},{row['molecule']}].junctions", row["junctions"], gold_row["junctions"])

    comp_gold = _rows_by_key(pl.read_csv(golden / "v10_tables" / "S5_mirror_junction_composition_ND.tsv", separator="\t"),
                              ["junction_molecules"])
    for row in summary["junction_composition"]:
        gold_row = comp_gold[row["junction_molecules"]]
        for k in ("reads", "median_molecules", "skera_pass_share"):
            _check(lines, ok, notes, sample, f"junction_composition[{row['junction_molecules']}].{k}", row[k], gold_row[k])

    size_gold = _rows_by_key(pl.read_csv(golden / "v10_tables" / "fig4c_mirror_array_size_rule_ND.tsv", separator="\t"), ["letter"])
    for row in summary["mirror_size_rule"]:
        gold_row = size_gold[row["letter"]]
        for k in ("reads", "predicted", "median_molecules", "share_matching"):
            _check(lines, ok, notes, sample, f"mirror_size_rule[{row['letter']}].{k}", row[k], gold_row[k])

    s10 = pl.read_csv(golden / "v10_tables" / "S10_mirror_molecules_written_by_skera_ND.tsv", separator="\t").filter(pl.col("set") == "nd_hash_skera26")
    s10_rows = _rows_by_key(s10, ["region", "class"])
    arms = summary["skera"]["mirror_arms"]
    for region in ("after", "before"):
        _check(lines, ok, notes, sample, f"skera.mirror_arms.{region}_clean_share", arms[f"{region}_clean_share"], s10_rows[(region, "CLEAN")]["share"])
        for cls in MIRROR_ARM_CLASSES:
            _check(lines, ok, notes, sample, f"skera.mirror_arms.{region}_counts[{cls}]", arms[f"{region}_counts"][cls], s10_rows[(region, cls)]["n"])


def _compare_sample(lines: list[str], ok: list[bool], notes: list[str], sample: str, summary: dict, golden: Path) -> None:
    _compare_top_level(lines, ok, notes, sample, summary, golden)
    _compare_patterns(lines, ok, notes, sample, summary, golden)
    _compare_skera(lines, ok, notes, sample, summary, golden)
    if sample == "BD144_ND":
        _compare_nd_only(lines, ok, notes, sample, summary, golden)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate kinnex-qc run's summary.json against the v10 study numbers")
    ap.add_argument("runs_dir", help="directory holding <sample>/summary.json for BD144_ND, BD144_HM, BD144_kin")
    ap.add_argument("--golden", default=str(Path(__file__).resolve().parent.parent / "tests" / "golden"),
                     help="golden tables directory (default: tests/golden)")
    args = ap.parse_args(argv)
    runs_dir, golden = Path(args.runs_dir), Path(args.golden)

    lines: list[str] = []
    ok: list[bool] = []
    notes: list[str] = []
    for sample in SAMPLES:
        summary = json.loads((runs_dir / sample / "summary.json").read_text())
        _compare_sample(lines, ok, notes, sample, summary, golden)

    print("\n".join(lines))
    if notes:
        print("\n" + "\n".join(notes))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
