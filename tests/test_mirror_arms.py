"""Tests for study/analyses/mirror_arms.py: what skera writes from single TSO-TSO mirror reads, on hand-built reads with
known answers, and the three v10 runs (hash sample with skera 26.1.0, calibration set with skera 26.1.0 and 1.2.0)
against the golden files."""
import gzip
import json

import polars as pl
import pysam
import pytest

from pipeline_helpers import GOLDEN_INPUTS, golden_diffs, run_analysis
from synth import complete_with_false_cut, double_mirror, frames, tso_tso_mirror

import mirror_arms as ma


def _mirror(letter: str, before: str, sreads: list) -> dict:
    """before' letter' [TSO-TSO] letter before: molecules 14-1014 (proper), 1031-3031 (TSO-TSO), 3048-4048 (proper)."""
    return {"adapters": [], "sreads": sreads,
            "molecules": [(14, 1014, "proper", False, f"{before}'", f"{letter}'"), (1031, 3031, "tso_tso", False, f"{letter}'", letter),
                          (3048, 4048, "proper", False, letter, before)]}


def _reads() -> dict:
    """zm 2: tso_tso_mirror at H, skera writes the arm after the junction only (AFTER_ONLY). zm 10 at D: an S-read
    covers all of the TSO-TSO molecule and half the molecule before it (SPANS_JUNCTION, descending). zm 11 at G: no
    S-read (NO_SREAD). zm 12 at F: one S-read per arm, the one before descending (BOTH_ARMS). zm 14 at C: one S-read
    covering 5% of the molecule after the junction (OTHER). zm 3: double_mirror, two junctions, not a single TSO-TSO
    mirror. zm 1 (complete_with_false_cut: FUSED, PARTIAL) and zm 20 (two CLEAN molecules): the complete arrays."""
    reads = {**complete_with_false_cut(), **tso_tso_mirror(), **double_mirror()}
    reads[10] = _mirror("D", "E", [(500, 3031, 4, 3)])
    reads[11] = _mirror("G", "H", [])
    reads[12] = _mirror("F", "G", [(14, 1014, 7, 6), (3048, 4048, 5, 6)])
    reads[14] = _mirror("C", "D", [(3040, 3100, 2, 3)])
    reads[20] = {"adapters": [], "molecules": [(17, 1017, "proper", False, "C", "D"), (1034, 2034, "proper", False, "D", "E")],
                 "sreads": [(17, 1017, 2, 3), (1034, 2034, 3, 4)]}
    return reads


# zm: (primary_pattern, skera_passing, read_len)
SUMMARY = {1: ("complete", True, 2051), 2: ("mirror", False, 4062), 3: ("mirror", False, 2068), 10: ("mirror", False, 4062),
           11: ("mirror", False, 4062), 12: ("mirror", True, 4062), 14: ("mirror", False, 4062), 20: ("complete", True, 2051)}


def _truth_dir(path):
    feats, mols, sreads = frames(_reads())
    path.mkdir()
    hifi = pl.lit("hifi").alias("source")
    feats.with_columns(hifi).write_parquet(path / "read_features.parquet")
    mols.with_columns(hifi).write_parquet(path / "annot_segments.parquet")
    sreads.write_parquet(path / "segments.parquet")
    pl.DataFrame([{"zm": zm, "source": "hifi", "primary_pattern": p, "skera_passing": s, "read_len": n} for zm, (p, s, n) in SUMMARY.items()],
                 schema_overrides={"zm": pl.UInt32, "read_len": pl.Int32}).write_parquet(path / "read_summary.parquet")
    return sreads


def _run(tmp_path, *extra):
    ma.main(["--truth", str(tmp_path / "t"), *extra, "--label", "toy", "--out", str(tmp_path / "out")])
    return json.loads((tmp_path / "out" / "toy_summary.json").read_text())


def _table(tmp_path, name):
    return pl.read_csv(tmp_path / "out" / f"toy_{name}.tsv", separator="\t")


def test_mirror_arms_on_hand_built_reads(tmp_path):
    _truth_dir(tmp_path / "t")
    summ = _run(tmp_path, "--segments", str(tmp_path / "t" / "segments.parquet"))
    assert summ == {"label": "toy", "single_tso_tso_mirror_reads": 5, "mirror_reads_non_passing": 4, "mirror_reads_non_passing_share": 0.8,
                    "proper_molecules_in_mirror_reads": 10, "clean_share_mirror_reads": 0.3, "complete_reads_baseline": 2,
                    "clean_share_complete_arrays": 0.5}
    assert _table(tmp_path, "molecules").rows() == [
        ("after", "CLEAN", 2, 0.4), ("after", "NOT_WRITTEN", 3, 0.6),
        ("before", "CLEAN", 1, 0.2), ("before", "NOT_WRITTEN", 3, 0.6), ("before", "PARTIAL", 1, 0.2),
        ("all_mirror_arms", "CLEAN", 3, 0.3), ("all_mirror_arms", "NOT_WRITTEN", 6, 0.6), ("all_mirror_arms", "PARTIAL", 1, 0.1),
        ("complete_array", "CLEAN", 2, 0.5), ("complete_array", "FUSED", 1, 0.25), ("complete_array", "PARTIAL", 1, 0.25)]
    assert _table(tmp_path, "reads").rows() == [("AFTER_ONLY", 1, 0.2), ("BOTH_ARMS", 1, 0.2), ("NO_SREAD", 1, 0.2), ("OTHER", 1, 0.2),
                                               ("SPANS_JUNCTION", 1, 0.2)]
    assert _table(tmp_path, "reads_by_letter").rows() == [("C", "OTHER", 1, 1.0), ("D", "SPANS_JUNCTION", 1, 1.0), ("F", "BOTH_ARMS", 1, 1.0),
                                                         ("G", "NO_SREAD", 1, 1.0), ("H", "AFTER_ONLY", 1, 1.0)]
    assert _table(tmp_path, "orientation").rows() == [("after", True, 2), ("before", False, 1), ("junction", False, 1)]
    assert not (tmp_path / "out" / "toy_found_adapters.tsv").exists()


def test_sreads_from_a_bam_non_passing_found_adapters_and_ligations(tmp_path):
    """The S-reads from an S-read BAM give the same tables as from segments.parquet. skera's non-passing BAM holds zm 2
    (full read length), 10 (full) and 11 (shorter). zm 2's adapters of ours: Q' and H' before the junction, H and Q after
    it; skera found H and Q, and one adapter where we have none; an entry of zm 1 (not a mirror read) is not counted."""
    sreads = _truth_dir(tmp_path / "t")
    reads = _reads()
    reads[2]["adapters"] = [("Q", "-", 0, 14, 0), ("H", "-", 1014, 1031, 0), ("H", "+", 3031, 3048, 0), ("Q", "+", 4048, 4062, 0)]
    feats, _, _ = frames(reads)
    feats.with_columns(pl.lit("hifi").alias("source")).write_parquet(tmp_path / "t" / "read_features.parquet")
    header = {"HD": {"VN": "1.6", "SO": "unknown"}, "RG": [{"ID": "rg1", "PL": "PACBIO", "DS": "READTYPE=SEGMENT"}]}
    with pysam.AlignmentFile(str(tmp_path / "sreads.bam"), "wb", header=header) as bam:
        for zm, _, qs, qe, dl, dr in sreads.rows():
            a = pysam.AlignedSegment()
            a.query_name, a.query_sequence, a.flag = f"movie/{zm}/ccs/{qs}_{qe}", "A" * (qe - qs), 4
            for tag, v in (("zm", zm), ("qs", qs), ("qe", qe), ("dl", dl), ("dr", dr)):
                a.set_tag(tag, v, value_type="i")
            bam.write(a)
    with pysam.AlignmentFile(str(tmp_path / "np.bam"), "wb", header=header) as bam:
        for zm, n in ((2, 4062), (10, 4062), (11, 100)):
            a = pysam.AlignedSegment()
            a.query_name, a.query_sequence, a.flag = f"movie/{zm}/ccs", "A" * n, 4
            a.set_tag("zm", zm, value_type="i")
            bam.write(a)
    with gzip.open(tmp_path / "found.csv.gz", "wt") as fh:
        fh.write("read,strand,adapter_index,start,end\n")
        for zm, ai, s, e in ((2, 7, 3031, 3048), (2, 8, 4050, 4062), (2, 3, 2000, 2017), (1, 2, 1017, 1034)):
            fh.write(f"movie/{zm}/ccs,+,{ai},{s},{e}\n")
    (tmp_path / "lig.csv").write_text("adapter_1,adapter_2,ligations\nA,B,10\nB,B,3\nC,D,5\n")
    summ = _run(tmp_path, "--sreads", str(tmp_path / "sreads.bam"), "--non-passing", str(tmp_path / "np.bam"),
                "--found-adapters", str(tmp_path / "found.csv.gz"), "--ligations", str(tmp_path / "lig.csv"))
    assert summ == {"label": "toy", "single_tso_tso_mirror_reads": 5, "non_passing_records": 3, "non_passing_records_full_read_length": 2,
                    "mirror_reads_non_passing": 3, "mirror_reads_non_passing_share": 0.6, "proper_molecules_in_mirror_reads": 10,
                    "clean_share_mirror_reads": 0.3, "complete_reads_baseline": 2, "clean_share_complete_arrays": 0.5,
                    "skera_found_entries": 3, "skera_found_at_none_of_our_adapters": 1, "ligations_same_letter": 3, "ligations_total": 18}
    assert _table(tmp_path, "found_adapters").rows() == [("after_arm", 1, 1.0), ("before_arm", 1, 0.0), ("junction_forward", 1, 1.0),
                                                        ("junction_reverse", 1, 0.0)]
    assert _table(tmp_path, "orientation").rows() == [("after", True, 2), ("before", False, 1), ("junction", False, 1)]


# The three v10 runs: label -> arguments, with "cal" and "hash" standing for the calibration directory and the hash sample's ND
RUNS = {
    "nd_hash_skera26": ["--truth", "hash", "--segments", "hash/segments.parquet", "--baseline-fraction", "0.01"],
    "nd_cal_skera26": ["--truth", "cal/nd_first200k_truth", "--segments", "cal/nd_first200k_truth/segments.parquet"],
    "nd_cal_skera12": ["--truth", "cal/nd_first200k_truth", "--sreads", "cal/nd_first200k.skera120.bam",
                       "--non-passing", "cal/nd_first200k.skera120.non_passing.bam",
                       "--found-adapters", "cal/nd_first200k.skera120.found_adapters.csv.gz",
                       "--ligations", "cal/nd_first200k.skera120.ligations.csv"],
}


@pytest.mark.real
@pytest.mark.slow
@pytest.mark.parametrize("label", RUNS)
def test_v10_runs_equal_golden(local, tmp_path, label):
    roots = {"cal": str(local["calibration_dir"]), "hash": str(local["hash_dir"] / "BD144_ND")}
    args = [a if a.split("/")[0] not in roots else a.replace(a.split("/")[0], roots[a.split("/")[0]], 1) for a in RUNS[label]]
    run_analysis("mirror_arms.py", *args, "--label", label, "--out", str(tmp_path / "out"))
    golden = GOLDEN_INPUTS / "skera_mirror_arms"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(p.name for p in golden.glob(f"{label}_*"))
    assert golden_diffs(tmp_path / "out", golden, tmp_path / "cmp", f"{label}_*") == []
