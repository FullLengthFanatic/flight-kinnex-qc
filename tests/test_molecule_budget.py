"""Tests for study/analyses/molecule_budget.py: the slot budget on hand-built reads with known answers, and the outputs
on the v10 hash sample against the golden files."""
import json

import polars as pl
import pytest

from pipeline_helpers import GOLDEN_INPUTS, golden_diffs, run_analysis
from synth import complete_with_false_cut, double_mirror, frames, tso_tso_mirror

import molecule_budget as mb

SAMPLES = ("BD144_ND", "BD144_HM", "BD144_kin")
PATTERNS = {1: "complete", 2: "mirror", 3: "mirror", 4: "truncated", 5: "complete"}
# (zm, di): lima_pass, refine_asrun, mapped_fltnc
FATES = {(1, 0): (True, True, True), (1, 1): (True, True, False), (2, 0): (True, True, True),
         (5, 0): (False, False, False), (5, 1): (True, False, False), (5, 2): (True, True, False), (5, 3): (True, True, True),
         (5, 4): (True, True, True)}


def _reads() -> dict:
    """zm 1: complete_with_false_cut, its two proper molecules FUSED and PARTIAL (S-read 0, mapped, covers both).
    zm 2: tso_tso_mirror, the H-Q molecule CLEAN and kept, Q'-H' not written. zm 3: double_mirror, no S-reads.
    zm 4: no molecule. zm 5: four proper molecules each written cleanly, lost at lima, at refine, at mapping and kept;
    a TSO-TSO molecule whose own S-read is mapped (an S-read of no usable molecule); a fused proper molecule."""
    reads = {**complete_with_false_cut(), **tso_tso_mirror(), **double_mirror()}
    reads[4] = {"adapters": [], "molecules": [], "sreads": []}
    reads[5] = {"adapters": [],
                "molecules": [(0, 1000, "proper", False, "A", "B"), (1000, 2000, "proper", False, "B", "C"),
                              (2000, 3000, "proper", False, "C", "D"), (3000, 4000, "proper", False, "D", "E"),
                              (4000, 5000, "tso_tso", False, "E", "F"), (5000, 6000, "proper", True, "F", "G")],
                "sreads": [(0, 1000, 0, 1), (1000, 2000, 1, 2), (2000, 3000, 2, 3), (3000, 4000, 3, 4), (4000, 5000, 4, 5)]}
    return reads


def _sample_dir(path, reads):
    _, mols, sreads = frames(reads)
    path.mkdir()
    pl.DataFrame({"zm": list(PATTERNS), "source": "hifi", "primary_pattern": list(PATTERNS.values())},
                 schema_overrides={"zm": pl.UInt32}).write_parquet(path / "read_summary.parquet")
    mols.with_columns(pl.lit("hifi").alias("source")).write_parquet(path / "annot_segments.parquet")
    sreads.write_parquet(path / "segments.parquet")
    pl.DataFrame([{"zm": zm, "di": di, "lima_pass": f[0], "refine_asrun": f[1], "mapped_fltnc": f[2]} for (zm, di), f in FATES.items()],
                 schema_overrides={"zm": pl.UInt32, "di": pl.Int16}).write_parquet(path / "fate.parquet")


def test_budget_on_hand_built_reads(tmp_path):
    _sample_dir(tmp_path / "s", _reads())
    mb.main(["--sample-dir", str(tmp_path / "s"), "--label", "toy", "--out", str(tmp_path / "out")])
    summ = json.loads((tmp_path / "out" / "toy_budget.json").read_text())
    per_read = {"molecules": [2, 3, 4, 0, 6], "artifact_tso_tso": [0, 1, 1, 0, 1], "artifact_r1_r1": [0, 0, 1, 0, 0],
                "artifact_one_primer_or_none": [0, 0, 0, 0, 0], "artifact_fused": [0, 0, 0, 0, 1], "usable": [2, 2, 2, 0, 4],
                "kept_standard": [0, 1, 0, 0, 1], "lost_lima": [0, 0, 0, 0, 1], "lost_refine": [0, 0, 0, 0, 1],
                "lost_mapping": [0, 0, 0, 0, 1], "not_written_cleanly": [2, 1, 2, 0, 0], "other_mapped": [1, 0, 0, 0, 1]}
    want = {"label": "toy", "hifi_reads": 5, "molecules": 3.0, "unfilled_slots": 5.0,
            **{k: sum(v) / 5 for k, v in per_read.items() if k != "molecules"}, "mapped_sreads": 4 / 5}
    assert list(summ) == ["label", "hifi_reads", "molecules", "unfilled_slots", "artifact_tso_tso", "artifact_r1_r1",
                          "artifact_one_primer_or_none", "artifact_fused", "usable", "kept_standard", "lost_lima", "lost_refine",
                          "lost_mapping", "not_written_cleanly", "other_mapped", "mapped_sreads"]
    assert summ == pytest.approx(want, abs=0)
    by_pattern = pl.read_csv(tmp_path / "out" / "toy_budget_by_pattern.tsv", separator="\t")
    assert by_pattern.columns == ["primary_pattern", "reads", *list(summ)[2:-1]]
    assert by_pattern["primary_pattern"].to_list() == ["complete", "mirror", "truncated"]
    complete = by_pattern.row(0, named=True)
    assert (complete["reads"], complete["molecules"], complete["unfilled_slots"], complete["kept_standard"], complete["other_mapped"]) \
        == (2, 4.0, 4.0, 0.5, 1.0)


@pytest.mark.real
@pytest.mark.slow
@pytest.mark.parametrize("sample", SAMPLES)
def test_hash_sample_equals_golden(local, tmp_path, sample):
    run_analysis("molecule_budget.py", "--sample-dir", str(local["hash_dir"] / sample), "--label", sample, "--out", str(tmp_path / "out"))
    golden = GOLDEN_INPUTS / "molecule_budget"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(p.name for p in golden.glob(f"{sample}_*"))
    assert golden_diffs(tmp_path / "out", golden, tmp_path / "cmp", f"{sample}_*") == []
