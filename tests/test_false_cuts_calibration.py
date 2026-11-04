"""Tests for study/analyses/false_cuts_calibration.py: the cuts skera makes without an adapter of ours on hand-built
reads with known sequences, and the three tables on the calibration set against the v10 golden files."""
import random

import polars as pl
import pysam
import pytest

from kinnex_qc.patterns import MAS, revcomp
from pipeline_helpers import GOLDEN_INPUTS, golden_diffs, run_analysis
from synth import complete_with_false_cut, frames

import false_cuts_calibration as fcc

U32 = {"zm": pl.UInt32}


def _mutated_b() -> str:
    """Adapter B with substitutions at 2, 8 and 14: 3 edits from B, at least 5 from every other adapter on either strand."""
    b = list(MAS["B"])
    for i in (2, 8, 14):
        b[i] = {"A": "C", "C": "G", "G": "T", "T": "A"}[b[i]]
    return "".join(b)


def _reads() -> tuple[dict, dict]:
    """zm 1: complete_with_false_cut (adapter D of ours inside the first S-read, an R1 hit 2 bp after it), the cut at
    1500-1514 holds adapter D itself, which carries a CTCCTCC repeat. zm 2: passing, one S-read with adapter D inside it
    and no cut. zm 3: not passing, a cut. zm 4: passing, a cut at 1200-1224 holding a 3-edit copy of B; a fail-read MAS
    hit in that gap is not ours (only HiFi hits count). Returns (reads for synth.frames, sequences by zm)."""
    rnd = random.Random(5)

    def seq(n, inserts):
        s = list("".join(rnd.choice("ACGT") for _ in range(n)))
        for pos, text in inserts:
            s[pos:pos + len(text)] = text
        return "".join(s)

    reads = complete_with_false_cut()
    reads[1]["adapters"].append(("R1", "+", 1036, 1060, 0))
    reads[2] = {"adapters": [("C", "+", 0, 17, 0), ("D", "+", 1017, 1034, 1)], "molecules": [], "sreads": [(17, 2000, 2, 3)]}
    reads[3] = {"adapters": [("A", "+", 0, 17, 0)], "molecules": [], "sreads": [(17, 900, 0, 1), (920, 1800, 1, 2)]}
    reads[4] = {"adapters": [("A", "+", 0, 17, 0)], "molecules": [], "sreads": [(17, 1200, 4, 5), (1224, 2900, 5, 6)]}
    seqs = {1: seq(2100, [(0, MAS["C"]), (1017, MAS["D"]), (1497, MAS["D"]), (2034, MAS["E"])]),
            2: seq(2100, [(0, MAS["C"]), (1017, MAS["D"])]),
            3: seq(1900, [(0, MAS["A"])]),
            4: seq(3000, [(0, MAS["A"]), (1200, _mutated_b())])}
    return reads, seqs


def _truth_dir(path, reads, passing):
    feats, _, sreads = frames(reads)
    feats = feats.with_columns(pl.when(pl.col("name") == "R1").then(pl.lit("R1")).otherwise(pl.col("kind")).alias("kind"),
                               pl.lit("hifi").alias("source"))
    fail_hit = pl.DataFrame([{"zm": 4, "kind": "MAS", "name": "F", "strand": "+", "start": 1203, "end": 1220, "edits": 0,
                              "source": "fail_sub"}], schema=feats.schema)
    path.mkdir()
    pl.concat([feats, fail_hit]).write_parquet(path / "read_features.parquet")
    sreads.write_parquet(path / "segments.parquet")
    pl.DataFrame({"zm": list(passing), "source": ["hifi"] * len(passing), "skera_passing": list(passing.values())},
                 schema_overrides=U32).write_parquet(path / "read_summary.parquet")


def _hifi_bam(path, seqs):
    header = {"HD": {"VN": "1.6", "SO": "unknown"}, "RG": [{"ID": "rg1", "PL": "PACBIO", "DS": "READTYPE=CCS"}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for zm, s in seqs.items():
            a = pysam.AlignedSegment()
            a.query_name, a.query_sequence, a.flag = f"movie/{zm}/ccs", s, 4
            a.set_tag("zm", zm, value_type="i")
            bam.write(a)


def test_closest_adapter():
    assert fcc.closest_adapter("TTTT" + MAS["D"] + "GG") == ("D", 0)
    assert fcc.closest_adapter(revcomp(MAS["G"])) == ("G", 0)
    assert fcc.closest_adapter(_mutated_b()) == ("B", 3)


def test_tables_on_hand_built_reads(tmp_path):
    reads, seqs = _reads()
    _truth_dir(tmp_path / "truth", reads, {1: True, 2: True, 3: False, 4: True})
    _hifi_bam(tmp_path / "hifi.bam", seqs)
    fcc.main(["--truth", str(tmp_path / "truth"), "--hifi", str(tmp_path / "hifi.bam"), "--out", str(tmp_path / "out")])
    summary = pl.read_csv(tmp_path / "out" / "skera_false_cuts_summary.tsv", separator="\t").to_dicts()
    assert summary == [{"skera_passing_reads": 3, "reads_with_cut_without_adapter": 2, "cuts": 2, "cuts_called_D_by_skera": 1,
                        "cuts_closest_to_D": 1, "cuts_closest_to_D_3_to_5_edits": 0, "cuts_within_2_edits_of_any_adapter": 1,
                        "cuts_with_ctcctcc_repeat": 1, "our_adapters_inside_sreads": 1, "inside_le1_edit_with_primer": 1}]
    by_letter = pl.read_csv(tmp_path / "out" / "skera_false_cuts_by_letter.tsv", separator="\t")
    assert by_letter.rows() == [("D", 1), ("F", 1)]            # tied counts in letter order
    by_distance = pl.read_csv(tmp_path / "out" / "skera_false_cuts_by_distance.tsv", separator="\t")
    assert by_distance.columns == ["closest_adapter", "edits", "cuts"]
    assert by_distance.rows() == [("B", 3, 1), ("D", 0, 1)]


def test_inside_adapters_count_once_and_only_in_reads_with_a_cut(tmp_path):
    """zm 2's adapter D inside its one S-read is left out (no cut in that read); an adapter held by two overlapping
    S-reads of zm 1 counts once."""
    reads, seqs = _reads()
    reads[1]["sreads"].append((20, 1480, 2, 3))
    _truth_dir(tmp_path / "truth", reads, {1: True, 2: True, 3: False, 4: True})
    _hifi_bam(tmp_path / "hifi.bam", seqs)
    fcc.main(["--truth", str(tmp_path / "truth"), "--hifi", str(tmp_path / "hifi.bam"), "--out", str(tmp_path / "out")])
    summary = pl.read_csv(tmp_path / "out" / "skera_false_cuts_summary.tsv", separator="\t").row(0, named=True)
    assert summary["our_adapters_inside_sreads"] == 1


@pytest.mark.real
def test_calibration_set_equals_golden(local, tmp_path):
    cal = local["calibration_dir"]
    run_analysis("false_cuts_calibration.py", "--truth", str(cal / "nd_first200k_truth"),
                 "--hifi", str(cal / "nd_first200k.hifi.nomod.bam"), "--out", str(tmp_path / "out"))
    golden = GOLDEN_INPUTS / "skera_false_cuts"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(p.name for p in golden.iterdir())
    assert golden_diffs(tmp_path / "out", golden, tmp_path / "cmp") == []
