"""Tests for study/analyses/downstream_test.py: the molecules it picks and groups on the synthetic test BAM, the
scoring of lima, refine and mapping outputs (--score-only), and the full run on the calibration set against the v10
golden file."""
import polars as pl
import pysam
import pytest

from kinnex_qc import runner
from kinnex_qc.io import iter_bam
from kinnex_qc.patterns import KINNEX8_SC3
from pipeline_helpers import GOLDEN_INPUTS, REPO, golden_diffs, run_analysis

import downstream_test as dt

# the truth fixture calls runner.annotate, which forks its pool workers in-process, as in tests/test_molecules.py
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


@pytest.fixture
def truth(test_bams, tmp_path):
    """The test BAM's reads annotated by the package, as a truth directory (read_summary, annot_segments, segments)."""
    hifi, sreads = test_bams
    rs, mols, _ = runner.annotate(hifi, KINNEX8_SC3, 1.0, 1)
    t = tmp_path / "truth"
    t.mkdir()
    hifi_src = pl.lit("hifi").alias("source")
    rs.with_columns(hifi_src).write_parquet(t / "read_summary.parquet")
    mols.with_columns(hifi_src).write_parquet(t / "annot_segments.parquet")
    runner.read_sreads(sreads, threads=1).write_parquet(t / "segments.parquet")
    return t


def test_molecules_and_groups(test_bams, truth):
    """Complete reads 1 to 3: 8 proper molecules each. TSO-TSO mirrors 4 and 5 and the R1-R1 mirror 6: the proper
    molecule after the junction is the one skera wrote, the one before it was not written; the junction molecule is not
    proper, so it is left out."""
    hifi, sreads = test_bams
    chosen = dt.choose(hifi, truth, 3000)
    assert chosen == {1: "complete", 2: "complete", 3: "complete", 4: "mirror", 5: "mirror", 6: "mirror"}
    seg = dt.molecule_groups(truth, chosen)
    counts = dict(seg.group_by("group").len().rows())
    assert counts == {"complete reads": 24, "mirror reads, written by skera": 3, "mirror reads, not written by skera": 3}
    written = seg.filter(pl.col("group") == "mirror reads, written by skera").sort("zm")
    assert written["cover"].to_list() == [1.0, 1.0, 1.0]


def _unaligned(path, names, template):
    with pysam.AlignmentFile(str(template), "rb", check_sq=False) as t:
        header = t.header.to_dict()
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for n in names:
            a = pysam.AlignedSegment()
            a.query_name, a.query_sequence, a.flag = n, "ACGT", 4
            bam.write(a)


def _trimmed(name: str, left: int = 3, right: int = 7) -> str:
    """name movie/zm/ccs/s_e as lima and refine rename it: the coordinates trimmed inside the molecule."""
    head, se = name.rsplit("/", 1)
    s, e = (int(x) for x in se.split("_"))
    return f"{head}/{s + left}_{e - right}"


def test_score_only(test_bams, truth, tmp_path, monkeypatch):
    """lima passes every molecule but one complete one, refine also drops one written mirror molecule, and the mapping
    leaves one more complete molecule unmapped."""
    hifi, sreads = test_bams
    out = tmp_path / "out"
    out.mkdir()
    seg = dt.molecule_groups(truth, dt.choose(hifi, truth, 3000))
    group = dt.write_molecules_bam(hifi, seg, sreads, out / "molecules.bam")
    names = [r.query_name for r in iter_bam(out / "molecules.bam")]
    assert list(group) == names
    complete = [n for n in names if group[n] == "complete reads"]
    written = [n for n in names if group[n] == "mirror reads, written by skera"]
    lima = [n for n in names if n != complete[0]]
    refine = [n for n in lima if n != written[0]]
    _unaligned(out / dt.LIMA_FL, [_trimmed(n) for n in lima], sreads)
    _unaligned(out / "mol.flnc.bam", [_trimmed(n, 10, 30) for n in refine], sreads)
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100000}]}
    with pysam.AlignmentFile(str(out / "mol.flnc.sam"), "w", header=header) as sam:
        for n in refine:
            a = pysam.AlignedSegment(sam.header)
            a.query_name, a.query_sequence = _trimmed(n, 10, 30), "ACGT"
            if n == complete[1]:
                a.flag = 4
            else:
                a.flag, a.reference_id, a.reference_start, a.cigarstring, a.mapping_quality = 0, 0, 100, "4M", 60
            sam.write(a)
    monkeypatch.chdir(REPO)                 # main reads its defaults from study/study.toml
    dt.main(["--hifi", str(hifi), "--template", str(sreads), "--truth", str(truth), "--ref", "unused.fa", "--out", str(out),
             "--score-only"])
    summary = pl.read_csv(out / "downstream_summary.tsv", separator="\t")
    assert summary.columns == ["group", "molecules", "lima_5p3p", "refine", "mapped"]
    assert summary.rows() == [("complete reads", 24, 23 / 24, 23 / 24, 22 / 24),
                              ("mirror reads, not written by skera", 3, 1.0, 1.0, 1.0),
                              ("mirror reads, written by skera", 3, 1.0, 2 / 3, 2 / 3)]
    per_molecule = pl.read_csv(out / "downstream_molecules.tsv", separator="\t")
    assert per_molecule["molecule"].to_list() == names


@pytest.mark.real
@pytest.mark.slow
def test_calibration_set_equals_golden(local, tmp_path):
    """lima, isoseq refine and minimap2 on the molecules of 3,000 mirror and 3,000 complete reads of the calibration set,
    mapped to the GRCh38 primary assembly: the per-group pass rates equal v10's, and the run log names minimap2 2.30."""
    cal = local["calibration_dir"]
    extra = ["--minimap2", str(local["minimap2"])] if "minimap2" in local else []
    r = run_analysis("downstream_test.py", "--hifi", str(cal / "nd_first200k.hifi.nomod.bam"), "--template", str(cal / "nd_first200k.skera120.bam"),
                     "--truth", str(cal / "nd_first200k_truth"), "--ref", str(local["reference_fasta"]), "--threads", "16",
                     "--pbtools-env", str(local["pbtools_env"]), *extra, "--out", str(tmp_path / "out"))
    assert "minimap2 2.30-r1287" in r.stderr
    assert "41,873 molecules from 5,839 reads" in r.stderr
    assert golden_diffs(tmp_path / "out", GOLDEN_INPUTS / "downstream_test", tmp_path / "cmp") == []
