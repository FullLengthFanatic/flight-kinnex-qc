"""Tests for study/calibration/: the adapter FASTA skera 1.2.0 gets, the calibration set built from synthetic BAMs, the
skera 1.2.0 call, the head test under both skera versions and the TSO-at-both-ends check on hand-built S-reads with
known answers, and, on this machine's calibration set, every output against v10."""
import array
import gzip
import importlib
import json
import random
import shutil
import subprocess

import polars as pl
import pysam
import pytest

from kinnex_qc.io import iter_bam
from kinnex_qc.patterns import MAS, MAS_ORDER, TSO_CORE, read_adapter_fasta, revcomp
from pipeline_helpers import CALIBRATION, GOLDEN_INPUTS, REPO, golden_diffs, run_script

import build_calibration_set as bcs
import head_version_check as hvc
import skera_split
import tso_end_validation as tev

# build_calibration_set annotates in a forked process pool in-process, as in tests/test_molecules.py
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")

GOLDEN = GOLDEN_INPUTS / "argentag" / "calibration"
HEADER = {"HD": {"VN": "1.6", "SO": "unknown"}, "RG": [{"ID": "rg1", "PL": "PACBIO", "DS": "READTYPE=CCS"}]}


def _write(path, records, header=HEADER):
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for r in records:
            bam.write(r)


def _record(zm, seq="ACGT", **tags):
    a = pysam.AlignedSegment()
    a.query_name, a.query_sequence, a.flag = f"movie/{zm}/ccs", seq, 4
    a.set_tag("zm", zm, value_type="i")
    for k, v in tags.items():
        a.set_tag(k, v, value_type="i")
    return a


def _sread(zm, qs, qe, dl, dr, seq="ACGT"):
    return _record(zm, seq, qs=qs, qe=qe, dl=dl, dr=dr)


def test_mas8_adapters_are_the_package_adapters():
    assert read_adapter_fasta(CALIBRATION / "mas8_adapters.fasta") == (MAS, tuple(MAS_ORDER))


def _samples(tmp_path, hifi, segmented, non_passing):
    table = tmp_path / "samples.tsv"
    cols = ["sample", "cell", "depletion", "hifi_bam", "fail_bam", "segmented_bam", "non_passing_bam", "zmw_metrics",
            "skera_report", "fltnc_bam"]
    row = ["BD144_ND", "cell", "none", str(hifi), "f.bam", str(segmented), str(non_passing), "z.csv.gz", "r.json", "a.bam"]
    table.write_text("\t".join(cols) + "\n" + "\t".join(row) + "\n")
    return table


def test_build_calibration_set(test_bams, tmp_path, monkeypatch):
    """The first 4 HiFi reads of the test BAM (N set to 4): their records without MM and ML; their skera S-reads (8 + 8
    + 8 + 1), read up to the first ZMW past the set (a later S-read of zm 2 is not taken); skera's verdict from the
    non-passing BAM, also read up to the first ZMW past the set (zm 2 and 4 non-passing, a later zm 3 record not
    taken); and the package's stage 02 annotation of the four reads."""
    hifi, sreads = test_bams
    with pysam.AlignmentFile(str(hifi), "rb", check_sq=False) as bam:
        header = bam.header.to_dict()
        reads = list(bam)
    for r in reads:
        r.set_tag("MM", "C+m?,0;", value_type="Z")
        r.set_tag("ML", array.array("B", [200]))
    _write(tmp_path / "hifi.bam", reads, header)
    seg, seen = [*iter_bam(sreads), _sread(2, 5, 50, 0, 1)], {}
    for r in seg:                                  # skera's di: the S-read's index within its ZMW
        zm = r.get_tag("zm")
        r.set_tag("di", seen.setdefault(zm, 0), value_type="i")
        seen[zm] += 1
    _write(tmp_path / "segmented.bam", seg)
    _write(tmp_path / "non_passing.bam", [_record(z) for z in (2, 4, 6, 3)])
    table = _samples(tmp_path, tmp_path / "hifi.bam", tmp_path / "segmented.bam", tmp_path / "non_passing.bam")
    monkeypatch.setattr(bcs, "N", 4)
    monkeypatch.chdir(REPO)
    out = tmp_path / "cal"
    bcs.main(["--samples", str(table), "--out", str(out), "--processes", "2"])

    nomod = list(iter_bam(out / "nd_first200k.hifi.nomod.bam"))
    assert [r.get_tag("zm") for r in nomod] == [1, 2, 3, 4]
    assert all(not r.has_tag("MM") and not r.has_tag("ML") and r.has_tag("np") and r.has_tag("rq") for r in nomod)
    assert [r.query_sequence for r in nomod] == [r.query_sequence for r in reads[:4]]

    truth = out / "nd_first200k_truth"
    segments = pl.read_parquet(truth / "segments.parquet")
    assert segments.schema == {"zm": pl.UInt32, "di": pl.Int64, "qs": pl.Int64, "qe": pl.Int64, "dl": pl.Int64, "dr": pl.Int64}
    want = [tuple(r.get_tag(t) for t in ("zm", "di", "qs", "qe", "dl", "dr")) for r in seg[:25]]
    assert segments.rows() == want and [z for z, *_ in want] == [1] * 8 + [2] * 8 + [3] * 8 + [4]
    rs = pl.read_parquet(truth / "read_summary.parquet")
    assert rs.select("zm", "source", "primary_pattern", "skera_passing").rows() == [
        (1, "hifi", "complete", True), (2, "hifi", "complete", False), (3, "hifi", "complete", True), (4, "hifi", "mirror", False)]
    assert sorted(p.name for p in truth.iterdir()) == ["annot_segments.parquet", "read_features.parquet", "read_summary.parquet",
                                                       "segments.parquet"]


def test_skera_split_command(tmp_path, monkeypatch, capsys):
    """skera 1.2.0 runs in the configured conda environment (a path: conda run -p; a name: -n) on the calibration BAM
    with the copied adapter FASTA, as v10 ran it; its version goes to stderr."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="skera 1.2.0\n\nUsing:\n  skera : 1.2.0\n", stderr="")

    monkeypatch.setattr(skera_split.subprocess, "run", run)
    monkeypatch.chdir(REPO)
    cal, out = tmp_path / "cal", tmp_path / "out"
    skera_split.main(["--calibration-dir", str(cal), "--out", str(out), "--skera-env", "/envs/skera120"])
    conda = ["conda", "run", "--no-capture-output", "-p", "/envs/skera120"]
    assert calls == [conda + ["skera", "--version"],
                     conda + ["skera", "split", "-j", "16", str(cal / "nd_first200k.hifi.nomod.bam"),
                              str(CALIBRATION / "mas8_adapters.fasta"), str(out / "nd_first200k.skera120.bam")]]
    assert "skera 1.2.0" in capsys.readouterr().err
    calls.clear()
    skera_split.main(["--calibration-dir", str(cal), "--skera-env", "skera120", "--threads", "4"])
    assert calls[1][:5] == ["conda", "run", "--no-capture-output", "-n", "skera120"]
    assert calls[1][-4:] == ["4", str(cal / "nd_first200k.hifi.nomod.bam"), str(CALIBRATION / "mas8_adapters.fasta"),
                             str(cal / "nd_first200k.skera120.bam")]


def _calibration_dir(tmp_path, sk120, sk26, patterns, hifi_zms):
    """A calibration folder: skera 1.2.0 S-reads (BAM), the HiFi ZMWs (nomod BAM), and in nd_first200k_truth the skera
    26.1.0 S-reads and our read calls."""
    d = tmp_path / "cal"
    (d / "nd_first200k_truth").mkdir(parents=True)
    _write(d / "nd_first200k.skera120.bam", [_sread(*s) for s in sk120])
    _write(d / "nd_first200k.hifi.nomod.bam", [_record(z) for z in hifi_zms])
    pl.DataFrame([(z, k, qs, qe, dl, dr) for k, (z, qs, qe, dl, dr) in enumerate(sk26)],
                 schema=["zm", "di", "qs", "qe", "dl", "dr"], orient="row").with_columns(pl.col("zm").cast(pl.UInt32)) \
        .write_parquet(d / "nd_first200k_truth" / "segments.parquet")
    pl.DataFrame({"zm": list(patterns), "source": ["hifi"] * len(patterns), "primary_pattern": list(patterns.values())},
                 schema_overrides={"zm": pl.UInt32}).write_parquet(d / "nd_first200k_truth" / "read_summary.parquet")
    return d


def test_head_version_check(tmp_path, monkeypatch):
    """Four HiFi ZMWs, zm 1 and 2 mirrors. skera 1.2.0: zm 1 flagged (head 3.048 molecules), zm 2 reaches Q but not A
    with a head of 1.5 molecules, zm 3 two S-reads from A, zm 4 flagged although complete. skera 26.1.0: zm 1 and 2
    flagged, zm 3 and 4 from A, and zm 9 (not among the HiFi ZMWs) left out."""
    sk120 = [(1, 3048, 4048, 7, 8), (2, 1500, 2500, 7, 8), (3, 17, 1000, 0, 1), (3, 1017, 2000, 1, 2), (4, 2500, 3500, 7, 8)]
    sk26 = [(1, 3048, 4048, 7, 8), (2, 3000, 4000, 7, 8), (3, 17, 1000, 0, 1), (3, 1017, 2000, 1, 2), (4, 17, 1000, 0, 1),
            (9, 3000, 4000, 7, 8)]
    d = _calibration_dir(tmp_path, sk120, sk26, {1: "mirror", 2: "mirror", 3: "complete", 4: "complete"}, [1, 2, 3, 4])
    monkeypatch.chdir(REPO)
    hvc.main(["--calibration-dir", str(d), "--out", str(tmp_path / "out")])
    t = pl.read_csv(tmp_path / "out" / "head_test_skera_versions.tsv", separator="\t")
    assert t.columns == ["skera", "hifi_zmws", "sreads", "q_not_a_share", "flagged", "flagged_share", "true_mirrors_flagged",
                         "ppv", "sensitivity"]
    assert t.rows() == [("skera 1.2.0", 4, 5, 0.75, 2, 0.5, 1, 0.5, 0.5), ("skera 26.1.0", 4, 5, 0.5, 2, 0.5, 2, 1.0, 1.0)]


def _seq(rnd, n, tso_head=False, tso_tail=False):
    s = list("".join(rnd.choice("ACGT") for _ in range(n)))
    if tso_head:
        s[10:10 + len(TSO_CORE)] = TSO_CORE
    if tso_tail:
        s[n - 40:n - 40 + len(TSO_CORE)] = revcomp(TSO_CORE)
    return "".join(s)


def test_tso_end_validation(tmp_path, monkeypatch):
    """Six S-reads, the first five checked (N set to 5). Flagged (a TSO core within 80 bp of each end): zm 1, overlapping
    zm 1's TSO-TSO molecule; zm 2 (1-1100), overlapping only a proper molecule; zm 3, a read without molecules; zm 4,
    whose TSO-TSO molecule lies outside the S-read. Not flagged: zm 2's second S-read (a TSO at the start only). The
    sixth S-read is flagged by sequence but past N."""
    rnd = random.Random(3)
    sk = [(1, 1000, 2000, True, True), (2, 100, 1100, True, True), (2, 1200, 2200, True, False), (3, 0, 500, True, True),
          (4, 5000, 6000, True, True), (5, 0, 1000, True, True)]
    d = tmp_path / "cal"
    (d / "nd_first200k_truth").mkdir(parents=True)
    _write(d / "nd_first200k.skera120.bam", [_record(z, _seq(rnd, qe - qs, h, t), qs=qs, qe=qe) for z, qs, qe, h, t in sk])
    mols = [(1, 0, 900, "proper"), (1, 900, 2100, "tso_tso"), (2, 0, 1200, "proper"), (4, 0, 1000, "tso_tso"),
            (4, 4900, 6100, "proper"), (5, 0, 1000, "tso_tso")]
    pl.DataFrame([(z, "hifi", s, e, c) for z, s, e, c in mols], schema=["zm", "source", "start", "end", "end_class"], orient="row") \
        .with_columns(pl.col("zm").cast(pl.UInt32), pl.col("start", "end").cast(pl.Int32)) \
        .write_parquet(d / "nd_first200k_truth" / "annot_segments.parquet")
    pl.DataFrame({"zm": [1, 2, 3, 4, 5], "source": ["hifi"] * 5, "primary_pattern": ["mirror", "complete", "unresolved", "mirror", "mirror"]},
                 schema_overrides={"zm": pl.UInt32}).write_parquet(d / "nd_first200k_truth" / "read_summary.parquet")
    monkeypatch.setattr(tev, "N", 5)
    monkeypatch.chdir(REPO)
    tev.main(["--calibration-dir", str(d), "--out", str(tmp_path / "out")])
    t = pl.read_csv(tmp_path / "out" / "tso_end_validation.tsv", separator="\t")
    assert t.columns == ["touches_tso_tso", "primary_pattern", "len", "share_of_sreads_checked", "sreads_checked"]
    assert sorted(t.rows()) == [(False, "complete", 1, 0.2, 5), (False, "mirror", 1, 0.2, 5), (False, "unresolved", 1, 0.2, 5),
                                (True, "mirror", 1, 0.2, 5)]


@pytest.mark.real
def test_head_version_check_equals_v10(local, tmp_path):
    run_script(CALIBRATION / "head_version_check.py", "--calibration-dir", str(local["calibration_dir"]), "--out", str(tmp_path / "out"))
    assert golden_diffs(tmp_path / "out", GOLDEN, tmp_path / "cmp", "head_test_skera_versions.tsv") == []


@pytest.mark.real
def test_tso_end_validation_equals_v10(local, tmp_path):
    run_script(CALIBRATION / "tso_end_validation.py", "--calibration-dir", str(local["calibration_dir"]), "--out", str(tmp_path / "out"))
    assert golden_diffs(tmp_path / "out", GOLDEN, tmp_path / "cmp", "tso_end_validation.tsv") == []


def _records(path):
    """(name, flag, sequence, qualities, tags with their type codes) of each record, in file order."""
    for r in iter_bam(path):
        yield r.query_name, r.flag, r.query_sequence, bytes(r.query_qualities), r.get_tags(with_value_type=True)


def _differing(a, b, fields: slice = slice(None)) -> tuple[int, int]:
    """(records, records that differ in fields) of two BAMs read side by side; fails if their record counts differ."""
    n = d = 0
    for x, y in zip(_records(a), _records(b), strict=True):
        n += 1
        d += x[fields] != y[fields]
    return n, d


def _through_sam_text(src, dest):
    """src's records after the trip v10's calibration BAM made (its @PG lines: samtools 1.13 view -h, then view -b)."""
    samtools = shutil.which("samtools")
    assert samtools, "samtools 1.13, the version of v10's @PG lines, must be on the PATH"
    version = subprocess.run([samtools, "--version"], capture_output=True, check=True).stdout.split()[1].decode()   # not all UTF-8
    assert version == "1.13", f"samtools {version}, not 1.13"
    view = subprocess.Popen([samtools, "view", "-h", str(src)], stdout=subprocess.PIPE)
    subprocess.run([samtools, "view", "-b", "-o", str(dest), "-"], stdin=view.stdout, check=True)
    view.stdout.close()
    assert view.wait() == 0


def _rq(path) -> pl.Series:
    return pl.Series([r.get_tag("rq") for r in iter_bam(path)], dtype=pl.Float32)


@pytest.mark.real
@pytest.mark.slow
def test_build_calibration_set_equals_v10(local, tmp_path):
    """From BD144_ND's plain-file inputs (study/samples.local.tsv). v10's nd_first200k.hifi.nomod.bam was not written by
    this script: its @PG lines say it was cut with samtools 1.13 through SAM text (view -h, then view -b, then view -x MM
    -x ML), and observed, that trip changes the float tags ec, rq and sn (6 significant digits in SAM text) and the type
    codes of integer tags. So the 200,000 records have v10's names, flags, sequences and qualities, and equal v10's
    records exactly, tags and type codes included, once they make the same trip with the same samtools. segments,
    annot_segments and read_features equal v10's; read_summary equals v10's except rq, which is each BAM's rq tag."""
    from compare_runs import compare_trees
    cal = local["calibration_dir"]
    r = run_script(CALIBRATION / "build_calibration_set.py", "--out", str(tmp_path), "--processes", "16")
    assert "200,000 reads, 30,741 non-passing, 1,461,865 skera 26.1.0 S-reads" in r.stderr
    new, v10_bam = tmp_path / "nd_first200k.hifi.nomod.bam", cal / "nd_first200k.hifi.nomod.bam"
    assert _differing(new, v10_bam, slice(4)) == (200_000, 0)
    _through_sam_text(new, tmp_path / "sam_text.bam")
    assert _differing(tmp_path / "sam_text.bam", v10_bam) == (200_000, 0)
    truth, v10 = tmp_path / "nd_first200k_truth", cal / "nd_first200k_truth"
    diffs = compare_trees(truth, v10, max_rows=1)
    assert len(diffs) == 1 and diffs[0].startswith("read_summary.parquet: row ")       # the other three parquets are equal
    rs, rs10 = pl.read_parquet(truth / "read_summary.parquet"), pl.read_parquet(v10 / "read_summary.parquet")
    assert rs.drop("rq").equals(rs10.drop("rq"))
    assert rs["rq"].equals(_rq(new)) and rs10["rq"].equals(_rq(tmp_path / "sam_text.bam"))
    assert (rs["rq"] != rs10["rq"]).sum() == 187_115


def _header_without_skera_cl(path) -> dict:
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        header = bam.header.to_dict()
    for pg in header["PG"]:
        if pg["ID"] == "skera-split":
            pg.pop("CL")
    return header


@pytest.mark.real
@pytest.mark.slow
def test_skera_split_equals_v10(local, tmp_path):
    """skera 1.2.0 from the configured environment on v10's calibration BAM: the S-read and non-passing BAMs equal v10's
    record for record (the header differs only in the command line skera records), the summary, ligations and read
    lengths byte for byte (summary.json but its run uuid), the found adapters once decompressed."""
    cal = local["calibration_dir"]
    r = run_script(CALIBRATION / "skera_split.py", "--calibration-dir", str(cal), "--out", str(tmp_path / "out"))
    assert r.stderr.splitlines()[0] == "skera 1.2.0"
    for name in ("nd_first200k.skera120.bam", "nd_first200k.skera120.non_passing.bam"):
        assert _differing(tmp_path / "out" / name, cal / name)[1] == 0, name
        headers = [_header_without_skera_cl(p) for p in (tmp_path / "out" / name, cal / name)]
        assert headers[0] == headers[1], name
    for name in ("summary.csv", "ligations.csv", "read_lengths.csv"):
        assert (tmp_path / "out" / f"nd_first200k.skera120.{name}").read_bytes() == (cal / f"nd_first200k.skera120.{name}").read_bytes()
    summary = [json.loads(p.read_text()) for p in (tmp_path / "out" / "nd_first200k.skera120.summary.json", cal / "nd_first200k.skera120.summary.json")]
    for s in summary:
        s.pop("uuid")
    assert summary[0] == summary[1]
    found = [gzip.decompress(p.read_bytes()) for p in (tmp_path / "out" / "nd_first200k.skera120.found_adapters.csv.gz",
                                                       cal / "nd_first200k.skera120.found_adapters.csv.gz")]
    assert found[0] == found[1]


@pytest.mark.parametrize("module", ["build_calibration_set", "skera_split", "head_version_check", "tso_end_validation"])
def test_calibration_dir_must_be_set(module, monkeypatch, capsys):
    """The public study.toml holds a placeholder: a script stops with a one-line error until the folder is given."""
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "load_config", lambda: {"calibration_dir": "<calibration>", "samples": "s.tsv", "processes": 2,
                                                     "skera120_env": "e"})
    with pytest.raises(SystemExit):
        mod.main([])
    flag = "--out" if module == "build_calibration_set" else "--calibration-dir"
    assert f"no calibration folder: give {flag} or set calibration_dir in study/study.local.toml" in capsys.readouterr().err
