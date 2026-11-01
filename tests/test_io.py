import pysam

from kinnex_qc.io import iter_bam


def _bam(path, zms):
    header = {"HD": {"VN": "1.6"}}
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i, zm in enumerate(zms):
            a = pysam.AlignedSegment()
            a.query_name = f"m/{zm}/ccs/{i}"
            a.query_sequence = "ACGT"
            a.flag = 4
            a.set_tag("zm", zm)
            out.write(a)


def test_iter_bam_all(tmp_path):
    _bam(tmp_path / "x.bam", [1, 1, 2, 3])
    assert [r.get_tag("zm") for r in iter_bam(tmp_path / "x.bam", threads=1)] == [1, 1, 2, 3]


def test_iter_bam_keep_reads_to_the_end(tmp_path):
    _bam(tmp_path / "x.bam", [1, 2, 2, 3, 4])
    got = [r.get_tag("zm") for r in iter_bam(tmp_path / "x.bam", threads=1, keep=lambda z: z % 2 == 0)]
    assert got == [2, 2, 4]


def test_iter_bam_max_records(tmp_path):
    _bam(tmp_path / "x.bam", [1, 2, 3, 4])
    assert len(list(iter_bam(tmp_path / "x.bam", threads=1, max_records=2))) == 2
