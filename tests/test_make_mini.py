"""Tests for study/make_mini.py on tiny plain-file inputs built with pysam."""
import gzip
import os
import subprocess
import sys
from pathlib import Path

import pysam

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "study" / "make_mini.py"
sys.path.insert(0, str(REPO / "study" / "pipeline"))

from config import EXPECTED_COLUMNS, load_samples  # noqa: E402

HEADER = {"HD": {"VN": "1.6", "SO": "unknown"}, "RG": [{"ID": "rg1", "PL": "PACBIO"}]}


def bam(path, zms, per_zm=1, mapped=False):
    header = dict(HEADER, SQ=[{"SN": "chr1", "LN": 1_000_000}]) if mapped else HEADER
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for zm in zms:
            for di in range(per_zm):
                a = pysam.AlignedSegment(out.header)
                a.query_name, a.query_sequence = f"m/{zm}/ccs/{di}", "ACGT" * 5
                if mapped:
                    a.reference_id, a.reference_start, a.cigarstring, a.flag = 0, 100 * zm, "20M", 0
                else:
                    a.flag = 4
                a.set_tags([("zm", zm), ("di", di), ("rq", 0.99876543, "f")])
                out.write(a)
    if mapped:
        pysam.sort("-o", str(path) + ".s.bam", str(path))
        os.replace(str(path) + ".s.bam", str(path))


def names(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as f:
        return [r.query_name for r in f]


def build_inputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    hifi_zms = list(range(1, 1601))                       # 1,600 HiFi ZMWs; the mini keeps the first 1,500
    bam(raw / "hifi.bam", hifi_zms)
    bam(raw / "seg.bam", [0] + hifi_zms + [3], per_zm=8)  # a leading ZMW outside the set is skipped; the trailing 3 is never reached
    bam(raw / "np.bam", hifi_zms[::50])
    bam(raw / "fail.bam", range(25_000, 26_200))          # the first 1,000 are kept; their metrics rows lie past row 20,000
    with gzip.open(raw / "zmw.csv.gz", "wt") as fh:
        fh.write("read_length,pol_length,read_qual,hifi,num_passes,zmw_id\n")
        for z in range(1, 30_001):
            fh.write(f"1000,5000,-1,1,0,{z}\n")
    (raw / "report.json").write_text('{"attributes": [{"id": "read_segmentation.reads", "value": 1600}]}\n')
    bam(raw / "fltnc.bam", hifi_zms, mapped=True)
    row = ["X", "C1", "none"] + [str(raw / f) for f in ("hifi.bam", "fail.bam", "seg.bam", "np.bam", "zmw.csv.gz", "report.json", "fltnc.bam")]
    (tmp_path / "samples.tsv").write_text("\t".join(EXPECTED_COLUMNS) + "\n" + "\t".join(row) + "\n")
    return raw


def test_make_mini(tmp_path):
    raw = build_inputs(tmp_path)
    mini = tmp_path / "mini"
    env = dict(os.environ, KINNEX_QC_MINI=str(mini))
    for _ in range(2):                                      # idempotent: the second run rewrites the same files and row
        subprocess.run([sys.executable, str(SCRIPT), "--samples", str(tmp_path / "samples.tsv")], env=env, cwd=REPO, check=True)

    samples = load_samples(mini / "samples_mini.tsv")
    assert list(samples) == ["X"]
    s = samples["X"]
    d = mini / "X"
    assert (s.cell, s.depletion) == ("C1", "none")
    assert (s.hifi_bam, s.fail_bam, s.zmw_metrics) == (d / "hifi_reads.bam", d / "fail_reads.bam", d / "zmw_metrics.csv.gz")
    assert (s.segmented_bam, s.non_passing_bam) == (d / "segmented.bam", d / "segmented.non_passing.bam")
    assert (s.skera_report, s.fltnc_bam) == (d / "read_segmentation.report.json", d / "X.fltnc.aligned.bam")

    assert names(s.hifi_bam) == [f"m/{z}/ccs/0" for z in range(1, 1501)]
    assert names(s.segmented_bam) == [f"m/{z}/ccs/{di}" for z in range(1, 1501) for di in range(8)]
    assert names(s.non_passing_bam) == [f"m/{z}/ccs/0" for z in range(1, 1501, 50)]
    assert names(s.fail_bam) == [f"m/{z}/ccs/0" for z in range(25_000, 26_000)]
    assert sorted(names(s.fltnc_bam)) == sorted(f"m/{z}/ccs/0" for z in range(1, 1501))
    with pysam.AlignmentFile(str(s.hifi_bam), "rb", check_sq=False) as f:
        assert f.header.to_dict()["RG"] == HEADER["RG"]
        assert next(iter(f)).get_tag("rq") == next(iter(pysam.AlignmentFile(str(raw / "hifi.bam"), "rb", check_sq=False))).get_tag("rq")

    lines = gzip.decompress(s.zmw_metrics.read_bytes()).decode().splitlines()
    all_lines = gzip.decompress((raw / "zmw.csv.gz").read_bytes()).decode().splitlines()
    assert lines == all_lines[:20_001] + all_lines[25_000:26_000]
    assert s.skera_report.read_bytes() == (raw / "report.json").read_bytes()
