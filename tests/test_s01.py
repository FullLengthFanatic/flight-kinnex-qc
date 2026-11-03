"""Tests for study/pipeline/s01_extract_segments.py."""
import polars as pl
import pysam
import pytest

from kinnex_qc.patterns import R1, TSO_CORE, TSO_EXT, revcomp
from kinnex_qc.sampling import in_hash_sample
from pipeline_helpers import bam_records, mini_samples, run_stage

from config import EXPECTED_COLUMNS

MOLECULE = R1 + "ACGTACGTA" + "GTGA" + "CATGCATGC" + "GACA" + "TTGACCAGT" + "AACCGGTT" + "T" * 20 + "GACT" * 150 + revcomp(TSO_CORE + TSO_EXT)


def write_bam(path, rows):
    header = {"HD": {"VN": "1.6", "SO": "unknown"}}
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, seq, tags in rows:
            a = pysam.AlignedSegment(out.header)
            a.query_name, a.query_sequence, a.flag = name, seq, 4
            a.set_tags(list(tags.items()))
            out.write(a)


@pytest.fixture
def fake_sample(tmp_path):
    """One ZMW read forward (dl<dr) and one read reversed (dl>dr), as plain BAMs listed in a samples table."""
    rows = []
    for zm, direction in ((100, 1), (200, -1)):
        for i in range(8):
            dl, dr = (i, i + 1) if direction == 1 else (8 - i, 7 - i)
            seq = MOLECULE if i % 2 == 0 else revcomp(MOLECULE)
            tags = {"zm": zm, "di": i, "dl": dl, "dr": dr, "qs": i * 1000, "qe": i * 1000 + len(seq), "np": 9, "rq": 0.999}
            rows.append((f"m/{zm}/ccs/{i}", seq, tags))
    write_bam(tmp_path / "segmented.bam", rows)
    write_bam(tmp_path / "hifi.bam", [(f"m/{zm}/ccs", "ACGT" * 10, {"zm": zm, "np": 9, "rq": 0.999}) for zm in (100, 200)])
    vals = ["FAKE", "c", "none", str(tmp_path / "hifi.bam"), str(tmp_path / "f.bam"), str(tmp_path / "segmented.bam"),
            str(tmp_path / "np.bam"), str(tmp_path / "z.csv.gz"), str(tmp_path / "r.json"), str(tmp_path / "fl.bam")]
    (tmp_path / "samples.tsv").write_text("\t".join(EXPECTED_COLUMNS) + "\n" + "\t".join(vals) + "\n")
    return tmp_path


def test_extract_fake(fake_sample):
    out = fake_sample / "out"
    run_stage("s01_extract_segments.py", "FAKE", out, fake_sample / "samples.tsv", "--fraction", "1")
    seg = pl.read_parquet(out / "FAKE" / "segments.parquet")
    assert seg.height == 16
    assert set(seg["end_class"]) == {"proper"}
    rev = seg.filter(pl.col("zm") == 200)
    assert rev["direction"].unique().to_list() == [-1]
    assert sorted(rev["pos"].to_list()) == list(range(8))
    fwd = seg.filter(pl.col("zm") == 100)
    assert fwd.filter(pl.col("di") == 0)["tso_toward_lower"].item() is False   # TSO in the tail, dl < dr
    assert fwd.filter(pl.col("di") == 1)["tso_toward_lower"].item() is True
    assert seg["insert_len"].unique().to_list() == [20 + 600 + 5]
    assert seg["internal_hits"].to_list() == [""] * 16
    assert all(s.startswith("R1:+") or s.startswith("TSO:+") for s in seg["head_hits"])


def test_fraction_keeps_the_hash_sample(fake_sample):
    """At --fraction 0.5 ZMW 100 is in the hash sample and ZMW 200 is not; outputs go under hash_0.5/."""
    assert in_hash_sample(100, 0.5) and not in_hash_sample(200, 0.5)
    out = fake_sample / "out"
    run_stage("s01_extract_segments.py", "FAKE", out, fake_sample / "samples.tsv", "--fraction", "0.5")
    seg = pl.read_parquet(out / "hash_0.5" / "FAKE" / "segments.parquet")
    assert seg["zm"].unique().to_list() == [100]
    assert seg.height == 8
    assert not (out / "FAKE").exists()


@pytest.mark.real
def test_extract_mini_nd(mini, tmp_path):
    run_stage("s01_extract_segments.py", "BD144_ND", tmp_path, mini / "samples_mini.tsv", "--fraction", "1")
    seg = pl.read_parquet(tmp_path / "BD144_ND" / "segments.parquet")
    assert seg.height == len(bam_records(mini_samples(mini)["BD144_ND"].segmented_bam))
    assert (seg["end_class"] == "proper").mean() > 0.85
    assert seg.filter(pl.col("end_class") != "proper")["insert_len"].null_count() == seg.filter(pl.col("end_class") != "proper").height
    assert ((seg["pos"] >= 0) & (seg["pos"] <= 7)).all()
