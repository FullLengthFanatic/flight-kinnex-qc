"""validate_reader.py against the real BD144 kinnex-qc run output, and against the tracked golden
tables directly for the comparison logic itself (no real run needed)."""
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "study"))

from validate_reader import KNOWN_DIFFS, SAMPLES, _check, _compare_nd_only, _compare_patterns, main  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden"


@pytest.mark.real
def test_validate_reader_matches_v10(local):
    validation_dir = local["validation_dir"]
    if not any((validation_dir / sample / "summary.json").is_file() for sample in SAMPLES):
        pytest.skip(f"no kinnex-qc run output under {validation_dir}; run the three BD144 validation runs first")
    assert main([str(validation_dir)]) == 0


def _nd_only_summary(golden: Path) -> dict:
    """An ND-only summary fragment (junctions, junction_composition, mirror_size_rule,
    skera.mirror_arms) built straight from the golden tables it is compared against, so it starts
    out matching them exactly. Tests perturb one value or drop one row from this baseline."""
    junctions = pl.read_csv(golden / "v10_tables" / "fig4b_mirror_junction_letters_ND.tsv", separator="\t").to_dicts()
    comp = pl.read_csv(golden / "v10_tables" / "S5_mirror_junction_composition_ND.tsv", separator="\t").to_dicts()
    size = pl.read_csv(golden / "v10_tables" / "fig4c_mirror_array_size_rule_ND.tsv", separator="\t").to_dicts()
    s10 = (pl.read_csv(golden / "v10_tables" / "S10_mirror_molecules_written_by_skera_ND.tsv", separator="\t")
           .filter(pl.col("set") == "nd_hash_skera26").to_dicts())
    arms = {}
    for region in ("after", "before"):
        arms[f"{region}_clean_share"] = next(r["share"] for r in s10 if r["region"] == region and r["class"] == "CLEAN")
        arms[f"{region}_counts"] = {r["class"]: r["n"] for r in s10 if r["region"] == region}
    return {
        "junctions": [{"letter": r["letter"], "molecule": r["molecule"], "junctions": r["junctions"]} for r in junctions],
        "junction_composition": [{"junction_molecules": r["junction_molecules"], "reads": r["reads"],
                                   "median_molecules": r["median_molecules"], "skera_pass_share": r["skera_pass_share"]}
                                  for r in comp],
        "mirror_size_rule": [{"letter": r["letter"], "reads": r["reads"], "predicted": r["predicted"],
                               "median_molecules": r["median_molecules"], "share_matching": r["share_matching"]}
                              for r in size],
        "skera": {"mirror_arms": arms},
    }


def test_compare_patterns_flags_a_row_the_reader_drops():
    gold = pl.read_csv(GOLDEN / "v10_tables" / "fig2a_chain_patterns.tsv", separator="\t").filter(pl.col("library") == "ND")
    rows = [{"pattern": r["pattern"], "reads": r["reads"], "share": r["share"]} for r in gold.to_dicts()]

    lines, ok = [], []
    _compare_patterns(lines, ok, "BD144_ND", {"patterns": rows}, GOLDEN)
    assert ok and all(ok)

    dropped = [row for row in rows if row["pattern"] != "other"]
    lines2, ok2 = [], []
    _compare_patterns(lines2, ok2, "BD144_ND", {"patterns": dropped}, GOLDEN)
    assert not all(ok2), "a pattern row missing from the reader's output must count as a DIFF"


def test_compare_nd_only_flags_a_row_the_reader_drops():
    summary = _nd_only_summary(GOLDEN)
    lines, ok = [], []
    _compare_nd_only(lines, ok, "BD144_ND", summary, GOLDEN)
    assert ok and all(ok)

    dropped = dict(summary, junctions=summary["junctions"][1:])
    lines2, ok2 = [], []
    _compare_nd_only(lines2, ok2, "BD144_ND", dropped, GOLDEN)
    assert not all(ok2), "a junctions row missing from the reader's output must count as a DIFF"


def test_check_prints_known_only_for_the_exact_pinned_values():
    (sample, key), (reader_val, v10_val, _reason) = next(iter(KNOWN_DIFFS.items()))
    lines, ok = [], []
    _check(lines, ok, sample, key, reader_val, v10_val)
    assert ok == [True]
    assert lines[0].endswith("\tKNOWN")


def test_check_flags_any_other_reader_value_on_a_known_diff_key_as_diff():
    (sample, key), (_reader_val, v10_val, _reason) = next(iter(KNOWN_DIFFS.items()))
    lines, ok = [], []
    _check(lines, ok, sample, key, 0.5, v10_val)
    assert ok == [False]
    assert lines[0].endswith("\tDIFF")


def test_compare_nd_only_flags_a_wrong_skera_pass_share():
    summary = _nd_only_summary(GOLDEN)
    lines, ok = [], []
    _compare_nd_only(lines, ok, "BD144_ND", summary, GOLDEN)
    assert ok and all(ok)

    tampered = dict(summary, junction_composition=[
        dict(row, skera_pass_share=0.5) if row["junction_molecules"] == "TSO-TSO only" else row
        for row in summary["junction_composition"]
    ])
    lines2, ok2 = [], []
    _compare_nd_only(lines2, ok2, "BD144_ND", tampered, GOLDEN)
    assert not all(ok2), "a wrong skera_pass_share must count as a DIFF, with no hidden exemption"
