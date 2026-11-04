"""Tests for study/argentag/: the S-read coordinates and the parent-read annotation on the synthetic test BAMs, the
full-read view on hand-built frames with known answers, the independent scan on the synthetic arrays (and that it
imports nothing from kinnex_qc), and, on this machine's ArgenTag data, every output against v10."""
import ast
import importlib
import shutil
import subprocess
import sys

import polars as pl
import pytest

from kinnex_qc import runner
from kinnex_qc.io import iter_bam
from pipeline_helpers import ARGENTAG, GOLDEN_INPUTS, REPO, golden_diffs, run_script
from synth import complete_with_false_cut, double_mirror, frames, tso_tso_mirror

import full_read_view as frv
import independent_scan as scan

GOLDEN = GOLDEN_INPUTS / "argentag" / "analysis"


def _argentag_dir(tmp_path, **files):
    """An ArgenTag folder as the scripts expect it: data/<name> for each BAM given."""
    d = tmp_path / "argentag"
    (d / "data").mkdir(parents=True)
    for name, src in files.items():
        shutil.copyfile(src, d / "data" / name)
    return d


def _files(d) -> list[str]:
    return sorted(str(p.relative_to(d)) for p in d.rglob("*"))


def test_independent_scan_imports_nothing_from_the_package():
    """Neither in its own import lines nor through the modules it imports (the study's config reader): imported in a
    fresh interpreter, it leaves kinnex_qc unloaded."""
    tree = ast.parse((ARGENTAG / "independent_scan.py").read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert names and not [m for m in names if m.split(".")[0] == "kinnex_qc"]
    code = f"import sys; sys.path.insert(0, {str(ARGENTAG)!r}); import independent_scan; print('kinnex_qc' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout == "False\n"


def test_sread_coords(test_bams, tmp_path):
    """zm, qs, qe, dl and dr of every S-read, in file order, with v10's dtypes, into --out; the ArgenTag folder is only
    read."""
    _, sreads = test_bams
    d = _argentag_dir(tmp_path, **{"segmented.bam": sreads})
    out = tmp_path / "out"
    run_script(ARGENTAG / "sread_coords.py", "--argentag-dir", str(d), "--out", str(out))
    assert _files(d) == ["data", "data/segmented.bam"]
    coords = pl.read_parquet(out / "argentag_sread_coords.parquet")
    assert coords.schema == {"zm": pl.Int64, "qs": pl.Int32, "qe": pl.Int32, "dl": pl.Int8, "dr": pl.Int8}
    want = [tuple(r.get_tag(t) for t in ("zm", "qs", "qe", "dl", "dr")) for r in iter_bam(sreads)]
    assert coords.rows() == want and len(want) == 3 * 8 + 3


def test_annotate_non_passing(test_bams, tmp_path):
    """Every read of the non-passing BAM annotated by the package's stage 02 function, source hifi and skera_passing
    false: the three complete arrays and the three mirrors (two TSO-TSO at H, one R1-R1 at B) of the test BAM."""
    hifi, _ = test_bams
    d = _argentag_dir(tmp_path, **{"segmented.non_passing.bam": hifi})
    out = tmp_path / "out"
    run_script(ARGENTAG / "annotate_non_passing.py", "--argentag-dir", str(d), "--out", str(out), "--processes", "2")
    assert _files(d) == ["data", "data/segmented.non_passing.bam"]
    rs = pl.read_parquet(out / "full_reads" / "read_summary.parquet")
    assert rs.select("zm", "source", "primary_pattern", "skera_passing").rows() == [
        (1, "hifi", "complete", False), (2, "hifi", "complete", False), (3, "hifi", "complete", False),
        (4, "hifi", "mirror", False), (5, "hifi", "mirror", False), (6, "hifi", "mirror", False)]
    items = [(r.get_tag("zm"), "hifi", r.query_sequence, r.get_tag("np"), r.get_tag("rq"), False) for r in iter_bam(hifi)]
    want = runner.annotate_batch(items)
    for name, cols in zip(("read_features", "read_summary", "annot_segments"), want):
        got = pl.read_parquet(out / "full_reads" / f"{name}.parquet")
        assert got.equals(pl.DataFrame(cols, schema=got.schema)), name


def _view_inputs():
    """Three non-passing reads (zm 1 complete, zm 2 a single TSO-TSO mirror at H, zm 3 a double mirror at B) and the
    S-reads of the cell: one for zm 2 (H..Q, 3,048 bp into the read, 1,000 bp long), none for zm 1 and 3, eight each for
    the passing reads 10 and 11."""
    _, seg, sreads = frames({**complete_with_false_cut(), **tso_tso_mirror(), **double_mirror()})
    rs = pl.DataFrame([(1, 12000, "C D E", "complete", 3), (2, 6000, "Q' H' H Q", "mirror", 4), (3, 2500, "A B B' B C", "mirror", 5)],
                      schema={"zm": pl.UInt32, "read_len": pl.Int32, "chain": pl.Utf8, "primary_pattern": pl.Utf8, "n_mas": pl.Int16},
                      orient="row")
    passing = [(zm, 17 + 1000 * k, 1000 * (k + 1), k, k + 1) for zm in (10, 11) for k in range(8)]
    coords = pl.concat([
        sreads.select(pl.col("zm").cast(pl.Int64), "qs", "qe", "dl", "dr"),
        pl.DataFrame(passing, schema={"zm": pl.Int64, "qs": pl.Int32, "qe": pl.Int32, "dl": pl.Int8, "dr": pl.Int8}, orient="row")])
    return rs, seg, coords


def test_full_read_view_tables():
    """By hand, with 5 HiFi ZMWs (3 non-passing, 2 passing): 9 molecules in non-passing reads (6 proper, 2 TSO-TSO, 1
    R1-R1) and 16 S-reads of passing reads, so 25 molecules in all; 3 mirror junctions (H TSO-TSO, B TSO-TSO, B R1-R1);
    zm 2 is the one single TSO-TSO mirror (3 molecules, as H predicts) and the one flagged ZMW (head 3.048 molecules);
    zm 3 is a mirror read without S-reads. Usable molecules per mirror read 2, S-reads per mirror read 0.5, per passing
    read 8."""
    rs, seg, coords = _view_inputs()
    t = frv.tables(rs, seg, coords, hifi=5)
    assert t["full_read_patterns"].rows() == [("mirror", 2, 4250.0, 0.4), ("complete", 1, 12000.0, 0.2)]
    assert t["full_read_junctions"].rows() == [("B", "R1-R1", 1), ("B", "TSO-TSO", 1), ("H", "TSO-TSO", 1)]
    assert t["full_read_mirror_rule"].rows() == [("H", 1, 3, 3.0, 1.0)]
    assert t["flag_validation"].rows(named=True) == [{
        "hifi_zmws": 5, "mirror_reads": 2, "flagged": 1, "flag_share": 0.2, "flag_ppv": 1.0, "flag_sensitivity": 0.5,
        "flag_non_mirror_share": 0.0, "q_not_a": 1, "q_not_a_ppv": 1.0, "mirror_without_sreads": 1}]
    assert t["full_read_summary"].rows(named=True) == [{
        "usable_per_mirror_read": 2.0, "sreads_per_mirror_read": 0.5, "sreads_per_passing_read": 8.0,
        "mirror_shortfall_ours_per_hifi_zmw": 2 / 5 * 6.0, "mirror_shortfall_sreads_per_hifi_zmw": 2 / 5 * 7.5,
        "proper_share_of_molecules": 6 / 9, "hifi_zmws": 5, "non_passing_reads": 3, "non_passing_share": 0.6,
        "mirror_reads": 2, "mirror_share": 0.4, "mirror_reads_median_len": 4250.0, "mirror_junctions": 3,
        "junctions_tso_tso_share": 2 / 3, "junctions_r1_r1_share": 1 / 3, "junctions_at_A_tso_tso": 0,
        "single_tt_reads": 1, "single_tt_matching_rule": 1.0, "molecules_denominator": 25, "molecules_in_non_passing": 9,
        "sreads_of_passing_reads": 16, "tso_tso_per_1000": 80.0, "r1_r1_per_1000": 40.0, "tso_tso_molecules": 2,
        "r1_r1_molecules": 1}]


def test_full_read_view_writes_the_five_tables(tmp_path):
    """The per-read tables of sread_coords.py and annotate_non_passing.py are read from --out, the run's ArgenTag
    folder, and the five tables go there."""
    rs, seg, coords = _view_inputs()
    out = tmp_path / "out"
    (out / "full_reads").mkdir(parents=True)
    rs.write_parquet(out / "full_reads" / "read_summary.parquet")
    seg.with_columns(pl.lit("hifi").alias("source")).write_parquet(out / "full_reads" / "annot_segments.parquet")
    coords.write_parquet(out / "argentag_sread_coords.parquet")
    frv.main(["--out", str(out)])
    names = sorted(p.name for p in out.iterdir() if p.is_file())
    assert names == ["argentag_sread_coords.parquet", "flag_validation.tsv", "full_read_junctions.tsv", "full_read_mirror_rule.tsv",
                     "full_read_patterns.tsv", "full_read_summary.tsv"]
    assert (out / "full_read_junctions.tsv").read_text() == "letter\tmolecule\tjunctions\nB\tR1-R1\t1\nB\tTSO-TSO\t1\nH\tTSO-TSO\t1\n"


@pytest.mark.parametrize("module", ["sread_coords", "annotate_non_passing", "independent_scan"])
def test_argentag_dir_must_be_set(module, tmp_path, monkeypatch, capsys):
    """The public study.toml holds a placeholder: a script stops with a one-line error until the folder is given."""
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "load_config", lambda: {"argentag_dir": "<argentag>", "processes": 2})
    with pytest.raises(SystemExit):
        mod.main(["--out", str(tmp_path)])
    assert "no ArgenTag folder: give --argentag-dir or set argentag_dir in study/study.local.toml" in capsys.readouterr().err


@pytest.mark.parametrize("module", ["sread_coords", "annotate_non_passing", "full_read_view", "independent_scan"])
def test_out_is_required(module, tmp_path, monkeypatch, capsys):
    """No script writes into the ArgenTag folder (inputs and v10's reference files) by default: --out has no default."""
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "load_config", lambda: {"argentag_dir": str(tmp_path), "processes": 2}, raising=False)
    with pytest.raises(SystemExit):
        mod.main([])
    assert "the following arguments are required: --out" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def _summary(path, patterns: dict[int, str]):
    pl.DataFrame({"zm": list(patterns), "primary_pattern": list(patterns.values())}, schema_overrides={"zm": pl.UInt32}) \
        .write_parquet(path)


def test_independent_scan_finds_the_arrays_of_the_test_bam(test_bams):
    """Nine adapters in order on the complete arrays, and a mirror call on the three mirrors (Q' H' H Q twice, A B B' A')."""
    hifi, _ = test_bams
    calls = {}
    for r in iter_bam(hifi):
        hits = scan.find_adapter_hits(r.query_sequence)
        calls[r.get_tag("zm")] = ("".join(h.adapter + ("'" if h.strand == "-" else "") for h in hits), scan.is_mirror_read(hits))
    assert calls == {1: ("ABCDEFGHQ", False), 2: ("ABCDEFGHQ", False), 3: ("ABCDEFGHQ", False),
                     4: ("Q'H'HQ", True), 5: ("Q'H'HQ", True), 6: ("ABB'A'", True)}


def test_independent_scan_output_holds_no_read_ids(test_bams, tmp_path):
    """On the six reads of the test BAM, with the pipeline's calls (read_summary of annotate_non_passing.py, in --out)
    changed for zm 1 (mirror) and zm 4 (complete): one disagreement each way. The text holds only aggregate lines; the
    two ZMWs go to the disagreement list, a private per-read table in --out, the run's ArgenTag folder. The ArgenTag
    folder is only read."""
    hifi, _ = test_bams
    d = _argentag_dir(tmp_path, **{"segmented.non_passing.bam": hifi})
    out = tmp_path / "out"
    (out / "full_reads").mkdir(parents=True)
    _summary(out / "full_reads" / "read_summary.parquet",
             {1: "mirror", 2: "complete", 3: "complete", 4: "complete", 5: "mirror", 6: "mirror"})
    r = run_script(ARGENTAG / "independent_scan.py", "--argentag-dir", str(d), "--out", str(out))
    text = (out / "independent_scan.txt").read_text()
    assert r.stdout == text
    assert text == """Reading BAM file...
Total records: 6
Sample size: 6
Scanning adapters in sample...
Loading reference results...

Concordance (sample of 6 reads):
  Same call (mirror or not): 4 (66.7%)
  Mirror in this scan only: 1 (16.7%)
  Mirror in the pipeline only: 1 (16.7%)

Disagreement details (first 10):

Mirror rate estimate:
  Sample: 3/6 = 50.00%
  Estimated mirror ZMWs in 6 non-passing: 3
  95% CI: [0, 5]
  95% CI as fraction: [9.99%, 90.01%]

Reference (in our sample ZMs):
  3/6 = 50.00%

Script executed successfully.
"""
    assert sorted(p.name for p in out.iterdir()) == ["full_reads", "independent_scan.txt", "independent_scan_disagreements.tsv"]
    assert _files(d) == ["data", "data/segmented.non_passing.bam"]
    dis = pl.read_csv(out / "independent_scan_disagreements.tsv", separator="\t")
    assert dis.columns == ["zm", "my_mirror", "their_mirror", "my_n_adapters"]
    assert sorted(dis.rows()) == [(1, False, True, 9), (4, True, False, 4)]


@pytest.mark.real
def test_full_read_view_equals_v10(local, tmp_path):
    """The five tables from v10's parent-read annotation and S-read coordinates (linked into the scratch output folder):
    the three frozen ones equal the golden files, the other two (aggregate tables the report does not read) equal the
    v10 files next to the inputs."""
    arg = local["argentag_dir"]
    out = tmp_path / "out"
    out.mkdir()
    for name in ("full_reads", "argentag_sread_coords.parquet"):
        (out / name).symlink_to(arg / "analysis" / name)
    run_script(ARGENTAG / "full_read_view.py", "--out", str(out))
    assert golden_diffs(out, GOLDEN, tmp_path / "cmp", "*.tsv") == []
    for name in ("full_read_patterns.tsv", "full_read_mirror_rule.tsv"):
        assert (out / name).read_bytes() == (arg / "analysis" / name).read_bytes(), name


@pytest.mark.real
def test_independent_scan_equals_v10(local, tmp_path):
    """From v10's parent-read annotation (linked into the scratch output folder): the aggregate text equals the golden
    file byte for byte, and the disagreement list next to it holds the eight ZMWs v10 printed, in the same order and
    with the same calls."""
    arg = local["argentag_dir"]
    out = tmp_path / "out"
    out.mkdir()
    (out / "full_reads").symlink_to(arg / "analysis" / "full_reads")
    run_script(ARGENTAG / "independent_scan.py", "--argentag-dir", str(arg), "--out", str(out))
    assert (out / "independent_scan.txt").read_bytes() == (GOLDEN / "independent_scan.txt").read_bytes()
    v10 = [line.split(". ", 1)[1] for line in (arg / "analysis" / "independent_scan.txt").read_text().splitlines()
           if line.startswith("  ") and ". ZM " in line]
    dis = pl.read_csv(out / "independent_scan_disagreements.tsv", separator="\t")
    assert [f"ZM {z}: my_mirror={m}, their={t}, n_adapters={n}" for z, m, t, n in dis.rows()] == v10 and len(v10) == 8


@pytest.mark.real
@pytest.mark.slow
def test_sread_coords_equals_v10(local, tmp_path):
    """One pass over the public S-read BAM (about 15 million records): the parquet equals v10's."""
    arg = local["argentag_dir"]
    run_script(ARGENTAG / "sread_coords.py", "--argentag-dir", str(arg), "--out", str(tmp_path))
    got = pl.read_parquet(tmp_path / "argentag_sread_coords.parquet")
    assert got.equals(pl.read_parquet(arg / "analysis" / "argentag_sread_coords.parquet"))


@pytest.mark.real
@pytest.mark.slow
def test_annotate_non_passing_equals_v10(local, tmp_path):
    """The 180,071 parent reads skera did not pass, annotated again: the three parquets equal v10's."""
    from compare_runs import compare_trees
    arg = local["argentag_dir"]
    run_script(ARGENTAG / "annotate_non_passing.py", "--argentag-dir", str(arg), "--out", str(tmp_path), "--processes", "16")
    assert compare_trees(tmp_path / "full_reads", arg / "analysis" / "full_reads") == []
