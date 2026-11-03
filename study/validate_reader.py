"""Validate `kinnex-qc run`'s summary.json against the frozen v10 study numbers, sample by sample.

Usage: study/validate_reader.py RUNS_DIR [--golden tests/golden]

RUNS_DIR/<sample>/summary.json for BD144_ND, BD144_HM and BD144_kin against the golden tables
and JSON files the study's own analyses wrote (tests/golden/v10_inputs, tests/golden/v10_tables).
Equality throughout is Python `==`, floats included: no tolerance, no rounding. Prints one line
per compared number (sample, key, the reader's value, v10's value, OK, KNOWN or DIFF), then one
line with the OK/KNOWN/DIFF counts, and exits 1 if any line is a DIFF, 0 when every line is OK or
KNOWN. A key present on one side only (the reader or v10) compares against None, so a row the
reader drops or adds is a DIFF too. See each `_compare_*` function below for exactly which
summary.json key compares against which golden file, and why.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import polars as pl

SAMPLES = ["BD144_ND", "BD144_HM", "BD144_kin"]
LIBRARY = {"BD144_ND": "ND", "BD144_HM": "HM", "BD144_kin": "kin"}

ARTIFACT_KEYS = ["proper_per_1000", "tso_tso_per_1000", "r1_r1_per_1000", "tso_only_per_1000",
                 "r1_only_per_1000", "none_per_1000", "other_per_1000", "internal_per_1000"]
# budget keys the reader and the golden fig6 table name the same; usable_not_clean (checked
# separately below) is the one budget number the golden table names not_written_cleanly instead.
BUDGET_KEYS = ["molecules", "unfilled_slots", "artifact_tso_tso", "artifact_r1_r1",
               "artifact_one_primer_or_none", "artifact_fused", "usable"]
MIRROR_ARM_CLASSES = ["CLEAN", "FUSED", "PARTIAL", "NOT_WRITTEN"]

# Known difference: kinnex-qc derives skera's pass or fail verdict from the S-read adapter labels
# (kinnex_qc.skera.passing_from_labels) because it reads only the segmented BAM, not skera's own
# non-passing BAM, and that disagrees with skera's own verdict on 20 of BD144_ND's 1,969,703
# sampled reads (19 truncated, 1 mirror, none of the complete arrays). The one mismatched mirror
# read falls in the row pinned below; see CHANGELOG.md for the exact numbers.
KNOWN_DIFFS = {
    ("BD144_ND", "junction_composition[TSO-TSO only].skera_pass_share"):
        (0.0034570434472299045, 0.0034602296623886418,
         "skera_pass_share derived from S-read labels, not skera's non-passing BAM"),
}


def _check(lines: list[str], ok: list[bool], sample: str, key: str, reader, v10) -> None:
    """Append one printed line (sample, key, reader's value, v10's value, OK/KNOWN/DIFF) and
    record whether it counts as a pass. `==` is exact for floats too: no rounding, no tolerance.
    A (sample, key) pinned in KNOWN_DIFFS prints KNOWN, and counts as a pass, only when both the
    reader's and v10's values equal the pinned ones exactly; any other value on that key, or a key
    not pinned at all, is OK when reader == v10 and DIFF otherwise."""
    pinned = KNOWN_DIFFS.get((sample, key))
    if pinned is not None and reader == pinned[0] and v10 == pinned[1]:
        status = "KNOWN"
    else:
        status = "OK" if reader == v10 else "DIFF"
    lines.append(f"{sample}\t{key}\t{reader!r}\t{v10!r}\t{status}")
    ok.append(status != "DIFF")


def _rows_by_key(df: pl.DataFrame, key_cols: list[str], none_as: str = "none") -> dict:
    """df's rows as a dict keyed by key_cols (a single value if key_cols has one column, else a
    tuple), a null in a key column standing for `none_as` (the false-cut class table uses an empty
    TSV cell for "no adapter letter")."""
    rows = {}
    for row in df.to_dicts():
        key = tuple(none_as if row[c] is None else row[c] for c in key_cols)
        rows[key[0] if len(key_cols) == 1 else key] = row
    return rows


def _compare_top_level(lines: list[str], ok: list[bool], sample: str, summary: dict, golden: Path) -> None:
    q2 = json.loads((golden / "v10_inputs" / "hash_0.18" / sample / "q2" / "q2_summary.json").read_text())
    _check(lines, ok, sample, "hifi_reads", summary["hifi_reads"], q2["hifi_reads"])
    _check(lines, ok, sample, "hifi_reads_in_file", summary["hifi_reads_in_file"], q2["hifi_reads_total"])
    for k in ("complete_fraction", "mirror_fraction"):
        _check(lines, ok, sample, f"fractions.{k}", summary["fractions"][k], q2[k])
    for k in ARTIFACT_KEYS:
        _check(lines, ok, sample, f"artifacts.{k}", summary["artifacts"][k], q2[k])
    for k in ("mirror_junctions", "mirror_junction_tso_tso_share", "mirror_junction_r1_r1_share"):
        _check(lines, ok, sample, f"mirror_junction_shares.{k}", summary["mirror_junction_shares"][k], q2[k])


def _compare_patterns(lines: list[str], ok: list[bool], sample: str, summary: dict, golden: Path) -> None:
    gold = pl.read_csv(golden / "v10_tables" / "fig2a_chain_patterns.tsv", separator="\t").filter(pl.col("library") == LIBRARY[sample])
    gold_by_pattern = _rows_by_key(gold, ["pattern"])
    reader_by_pattern = {row["pattern"]: row for row in summary["patterns"]}
    for pattern in sorted(set(reader_by_pattern) | set(gold_by_pattern)):
        reader_row = reader_by_pattern.get(pattern)
        gold_row = gold_by_pattern.get(pattern)
        for k in ("reads", "share"):
            got = reader_row[k] if reader_row is not None else None
            want = gold_row[k] if gold_row is not None else None
            _check(lines, ok, sample, f"patterns[{pattern}].{k}", got, want)


def _compare_skera(lines: list[str], ok: list[bool], sample: str, summary: dict, golden: Path) -> None:
    sk = summary["skera"]
    fc_gold = json.loads((golden / "v10_inputs" / "false_cut_anatomy" / f"{sample}_false_cuts.json").read_text())
    _check(lines, ok, sample, "skera.complete_passing_reads", sk["complete_passing_reads"], fc_gold["complete_passing_reads"])
    _check(lines, ok, sample, "skera.false_cuts", sk["false_cuts"], fc_gold["false_cuts"])
    _check(lines, ok, sample, "skera.false_cut_reads_share", sk["false_cut_reads_share"], fc_gold["reads_with_false_cut"])
    _check(lines, ok, sample, "skera.false_cuts_per_read", sk["false_cuts_per_read"], fc_gold["false_cuts_per_read"])

    classes = pl.read_csv(golden / "v10_inputs" / "false_cut_anatomy" / f"{sample}_false_cut_classes.tsv", separator="\t")
    letter_tables = {"false_cut_letters": _rows_by_key(classes.filter(pl.col("field") == "skera_letter"), ["value"]),
                      "skipped_adapter_letters": _rows_by_key(classes.filter(pl.col("field") == "missed"), ["value"])}
    for label, gold_rows in letter_tables.items():
        reader_dict = sk[label]
        for letter in sorted(set(reader_dict) | set(gold_rows)):
            got = reader_dict.get(letter)
            want = gold_rows[letter]["false_cuts"] if letter in gold_rows else None
            _check(lines, ok, sample, f"skera.{label}[{letter}]", got, want)

    budget_gold = pl.read_csv(golden / "v10_tables" / "fig6_molecule_budget.tsv", separator="\t").filter(pl.col("library") == LIBRARY[sample]).to_dicts()[0]
    for k in BUDGET_KEYS:
        _check(lines, ok, sample, f"skera.budget.{k}", sk["budget"][k], budget_gold[k])
    _check(lines, ok, sample, "skera.budget.usable_not_clean", sk["budget"]["usable_not_clean"], budget_gold["not_written_cleanly"])


def _compare_nd_only(lines: list[str], ok: list[bool], sample: str, summary: dict, golden: Path) -> None:
    junctions_gold = _rows_by_key(pl.read_csv(golden / "v10_tables" / "fig4b_mirror_junction_letters_ND.tsv", separator="\t"),
                                   ["letter", "molecule"])
    reader_junctions = {(row["letter"], row["molecule"]): row for row in summary["junctions"]}
    for key in sorted(set(reader_junctions) | set(junctions_gold)):
        reader_row = reader_junctions.get(key)
        gold_row = junctions_gold.get(key)
        got = reader_row["junctions"] if reader_row is not None else None
        want = gold_row["junctions"] if gold_row is not None else None
        _check(lines, ok, sample, f"junctions[{key[0]},{key[1]}].junctions", got, want)

    comp_gold = _rows_by_key(pl.read_csv(golden / "v10_tables" / "S5_mirror_junction_composition_ND.tsv", separator="\t"),
                              ["junction_molecules"])
    reader_comp = {row["junction_molecules"]: row for row in summary["junction_composition"]}
    for jm in sorted(set(reader_comp) | set(comp_gold)):
        reader_row = reader_comp.get(jm)
        gold_row = comp_gold.get(jm)
        for k in ("reads", "median_molecules", "skera_pass_share"):
            got = reader_row[k] if reader_row is not None else None
            want = gold_row[k] if gold_row is not None else None
            _check(lines, ok, sample, f"junction_composition[{jm}].{k}", got, want)

    size_gold = _rows_by_key(pl.read_csv(golden / "v10_tables" / "fig4c_mirror_array_size_rule_ND.tsv", separator="\t"), ["letter"])
    reader_size = {row["letter"]: row for row in summary["mirror_size_rule"]}
    for letter in sorted(set(reader_size) | set(size_gold)):
        reader_row = reader_size.get(letter)
        gold_row = size_gold.get(letter)
        for k in ("reads", "predicted", "median_molecules", "share_matching"):
            got = reader_row[k] if reader_row is not None else None
            want = gold_row[k] if gold_row is not None else None
            _check(lines, ok, sample, f"mirror_size_rule[{letter}].{k}", got, want)

    s10 = pl.read_csv(golden / "v10_tables" / "S10_mirror_molecules_written_by_skera_ND.tsv", separator="\t").filter(pl.col("set") == "nd_hash_skera26")
    s10_rows = _rows_by_key(s10, ["region", "class"])
    arms = summary["skera"]["mirror_arms"]
    for region in ("after", "before"):
        _check(lines, ok, sample, f"skera.mirror_arms.{region}_clean_share", arms[f"{region}_clean_share"], s10_rows[(region, "CLEAN")]["share"])
        for cls in MIRROR_ARM_CLASSES:
            _check(lines, ok, sample, f"skera.mirror_arms.{region}_counts[{cls}]", arms[f"{region}_counts"][cls], s10_rows[(region, cls)]["n"])


def _compare_sample(lines: list[str], ok: list[bool], sample: str, summary: dict, golden: Path) -> None:
    _compare_top_level(lines, ok, sample, summary, golden)
    _compare_patterns(lines, ok, sample, summary, golden)
    _compare_skera(lines, ok, sample, summary, golden)
    if sample == "BD144_ND":
        _compare_nd_only(lines, ok, sample, summary, golden)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate kinnex-qc run's summary.json against the v10 study numbers")
    ap.add_argument("runs_dir", help="directory holding <sample>/summary.json for BD144_ND, BD144_HM, BD144_kin")
    ap.add_argument("--golden", default=str(Path(__file__).resolve().parent.parent / "tests" / "golden"),
                     help="golden tables directory (default: tests/golden)")
    args = ap.parse_args(argv)
    runs_dir, golden = Path(args.runs_dir), Path(args.golden)

    lines: list[str] = []
    ok: list[bool] = []
    for sample in SAMPLES:
        summary = json.loads((runs_dir / sample / "summary.json").read_text())
        _compare_sample(lines, ok, sample, summary, golden)

    print("\n".join(lines))
    counts = Counter(line.rsplit("\t", 1)[1] for line in lines)
    print(f"OK={counts['OK']} KNOWN={counts['KNOWN']} DIFF={counts['DIFF']}")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
