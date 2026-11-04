"""Tests for study/analyses/false_cut_anatomy.py: the anatomy of skera's false cuts on hand-built reads with known
answers, and the outputs on the v10 hash sample against the golden files."""
import json

import polars as pl
import pytest

from pipeline_helpers import GOLDEN_INPUTS, golden_diffs, run_analysis
from synth import complete_with_false_cut, frames

import false_cut_anatomy as fca

SAMPLES = ("BD144_ND", "BD144_HM", "BD144_kin")


def _reads() -> dict:
    """zm 1: complete_with_false_cut, ascending (skera's dl < dr); the cut at 1500-1514 (centre 1507) lies in molecule
    D-E (1034-2034, R1 at its left end, TSO at its right end); our D (1017-1034) is left inside the first S-read, so
    the second S-read is the fragment, keeping the TSO end, and the first one's start sits at our C (0-17).
    zm 5: descending (dl > dr); the cut at 600-614 (centre 607) lies in molecule E'-D' (17-1017, R1 at its left end);
    our D' (1017-1034) is left inside the second S-read, so the first one is the fragment, keeping the R1 end, and the
    second one ends at our C' (2034-2051).
    zm 6: complete, passing, cut at its adapters only. zm 7: complete, not passing, a false cut. zm 8: truncated,
    passing, a false cut. Only zm 1, 5 and 6 are complete arrays skera passes."""
    reads = complete_with_false_cut()
    reads[1]["adapters"] += [("R1", "+", 1036, 1060, 0), ("TSO", "-", 2004, 2030, 0)]
    reads[5] = {"adapters": [("E", "-", 0, 17, 0), ("R1", "-", 20, 44, 0), ("D", "-", 1017, 1034, 0), ("C", "-", 2034, 2051, 0)],
                "molecules": [(17, 1017, "proper", False, "E'", "D'"), (1034, 2034, "proper", False, "D'", "C'")],
                "sreads": [(17, 600, 4, 3), (614, 2034, 3, 2)]}
    reads[6] = {"adapters": [("C", "+", 0, 17, 0), ("D", "+", 1017, 1034, 0), ("E", "+", 2034, 2051, 0)],
                "molecules": [(17, 1017, "proper", False, "C", "D"), (1034, 2034, "proper", False, "D", "E")],
                "sreads": [(17, 1017, 2, 3), (1034, 2034, 3, 4)]}
    reads[7] = {"adapters": [("C", "+", 0, 17, 0), ("D", "+", 1017, 1034, 0)],
                "molecules": [(17, 1017, "proper", False, "C", "D")], "sreads": [(17, 500, 2, 3), (514, 1017, 3, 4)]}
    reads[8] = {"adapters": [("C", "+", 0, 17, 0), ("D", "+", 1017, 1034, 0)],
                "molecules": [(17, 1017, "proper", False, "C", "D")], "sreads": [(17, 500, 2, 3), (514, 1017, 3, 4)]}
    return reads


SUMMARY = {1: ("complete", True), 5: ("complete", True), 6: ("complete", True), 7: ("complete", False), 8: ("truncated", True)}
# (zm, di): S-read end_class and fate (lima_pass, lima_pair, refine_asrun, mapped_fltnc)
FATES = {(1, 0): ("r1_only", True, "3p--3p", True, True), (1, 1): ("tso_only", False, "5p--5p", False, False),
         (5, 0): ("r1_only", False, "3p--3p", False, False), (5, 1): ("tso_only", True, "5p--5p", True, False),
         (6, 0): ("proper", True, "5p--3p", True, True), (6, 1): ("proper", True, "5p--3p", True, False)}


def _sample_dir(path, reads: dict, summary: dict = SUMMARY):
    feats, mols, sreads = frames(reads)
    feats = feats.with_columns(pl.when(pl.col("name").is_in(["TSO", "R1"])).then(pl.col("name")).otherwise(pl.col("kind")).alias("kind"))
    path.mkdir()
    hifi = pl.lit("hifi").alias("source")
    feats.with_columns(hifi).write_parquet(path / "read_features.parquet")
    mols.with_columns(hifi).write_parquet(path / "annot_segments.parquet")
    pl.DataFrame([{"zm": zm, "source": "hifi", "primary_pattern": p, "skera_passing": s} for zm, (p, s) in summary.items()],
                 schema_overrides={"zm": pl.UInt32}).write_parquet(path / "read_summary.parquet")
    cls = pl.DataFrame([{"zm": zm, "di": di, "end_class": f[0]} for (zm, di), f in FATES.items()],
                       schema_overrides={"zm": pl.UInt32, "di": pl.Int16})
    sreads.join(cls, on=["zm", "di"], how="left").with_columns(pl.col("end_class").fill_null("none")) \
        .write_parquet(path / "segments.parquet")
    pl.DataFrame([{"zm": zm, "di": di, "lima_pass": f[1], "lima_pair": f[2], "refine_asrun": f[3], "mapped_fltnc": f[4]}
                  for (zm, di), f in FATES.items()], schema_overrides={"zm": pl.UInt32, "di": pl.Int16}).write_parquet(path / "fate.parquet")


def test_anatomy_on_hand_built_reads(tmp_path):
    _sample_dir(tmp_path / "s", _reads())
    fca.main(["--sample-dir", str(tmp_path / "s"), "--label", "toy", "--out", str(tmp_path / "out")])
    summ = json.loads((tmp_path / "out" / "toy_false_cuts.json").read_text())
    assert summ == {
        "label": "toy", "complete_passing_reads": 3, "false_cuts": 2, "false_cuts_per_read": 2 / 3, "reads_with_false_cut": 2 / 3,
        "skera_calls_D": 1.0, "missed_adapter_found": 1.0, "missed_is_D": 1.0, "missed_same_letter_as_skera_call": 1.0,
        "cut_inside_proper_molecule": 1.0, "median_cut_to_missed_adapter_bp": 450.0,      # 490 (zm 1) and 410 (zm 5)
        "fused_far_end_at_real_adapter": 1.0, "fragment_after_cut_in_read_order": 0.5,
        "fragment_keeps_tso_when_ascending": 1.0, "fragment_keeps_r1_when_descending": 1.0,
        "cut_position_from_kept_end_median": pytest.approx((1 - 473 / 1000 + 590 / 1000) / 2),
        "fates": {"frag_lima": 0.0, "frag_refine": 0.0, "frag_mapped": 0.0, "fused_lima": 1.0, "fused_refine": 1.0, "fused_mapped": 0.5},
        "baseline_proper_sreads": {"lima_pass": 1.0, "refine_asrun": 1.0, "mapped_fltnc": 0.5}}
    classes = pl.read_csv(tmp_path / "out" / "toy_false_cut_classes.tsv", separator="\t")
    assert classes.columns == ["field", "value", "false_cuts"]
    assert sorted(classes.rows()) == sorted([
        ("skera_letter", "D", 2), ("missed", "D", 2), ("fragment", "S1", 1), ("fragment", "S2", 1),
        ("fragment_keeps", "TSO", 1), ("fragment_keeps", "R1", 1), ("frag_class", "tso_only", 1), ("frag_class", "r1_only", 1),
        ("fused_class", "r1_only", 1), ("fused_class", "tso_only", 1), ("frag_pair", "5p--5p", 1), ("frag_pair", "3p--3p", 1),
        ("fused_pair", "3p--3p", 1), ("fused_pair", "5p--5p", 1)])
    assert classes["field"].to_list() == sorted(classes["field"].to_list())


def test_cut_without_a_missed_adapter_has_no_fragment(tmp_path):
    """zm 9: zm 1 without our D, so its cut has no missed adapter and no fragment; it counts in the false cuts, not in
    the fates."""
    reads = _reads()
    reads[9] = {**reads[1], "adapters": [a for a in reads[1]["adapters"] if a[0] != "D"]}
    _sample_dir(tmp_path / "s", reads, {**SUMMARY, 9: ("complete", True)})
    fca.main(["--sample-dir", str(tmp_path / "s"), "--label", "toy", "--out", str(tmp_path / "out")])
    summ = json.loads((tmp_path / "out" / "toy_false_cuts.json").read_text())
    assert (summ["complete_passing_reads"], summ["false_cuts"]) == (4, 3)
    assert summ["missed_adapter_found"] == 2 / 3
    assert summ["fragment_after_cut_in_read_order"] == 0.5          # S2 (zm 1), S1 (zm 5); zm 9 has none
    assert summ["fates"] == {"frag_lima": 0.0, "frag_refine": 0.0, "frag_mapped": 0.0, "fused_lima": 1.0, "fused_refine": 1.0,
                             "fused_mapped": 0.5}
    classes = pl.read_csv(tmp_path / "out" / "toy_false_cut_classes.tsv", separator="\t")
    assert {("fragment", None, 1), ("missed", None, 1), ("missed", "D", 2)} <= set(classes.rows())


@pytest.mark.real
@pytest.mark.slow
@pytest.mark.parametrize("sample", SAMPLES)
def test_hash_sample_equals_golden(local, tmp_path, sample):
    run_analysis("false_cut_anatomy.py", "--sample-dir", str(local["hash_dir"] / sample), "--label", sample, "--out", str(tmp_path / "out"))
    golden = GOLDEN_INPUTS / "false_cut_anatomy"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(p.name for p in golden.glob(f"{sample}_*"))
    assert golden_diffs(tmp_path / "out", golden, tmp_path / "cmp", f"{sample}_*") == []
