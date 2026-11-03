"""Tests for `--write-molecules`: a BAM of the molecules a full-read segmentation finds."""
import subprocess
import sys
import tracemalloc
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import polars as pl
import pysam
import pytest

from kinnex_qc import molecules, runner
from kinnex_qc.patterns import KINNEX8_SC3

PRIMERS = Path(__file__).resolve().parent.parent / "resources" / "primers_FlashBD.fasta"

# every test in this module calls runner.annotate, which forks its pool workers in-process; same
# warning text as tests/test_cli_run.py's _IGNORE_FORK_WARNING
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


def _annotated(hifi):
    _, mols, _ = runner.annotate(hifi, KINNEX8_SC3, 1.0, 2)
    return mols


def test_write_molecules_count_matches_usable_molecules(test_bams, tmp_path):
    """8 usable molecules each for the three complete arrays (zm 1, 2, 3) and 2 each for the three
    mirrors (zm 4, 5, 6: the junction molecule itself is not usable): 24 + 6 = 30 (hand-derived in
    tests/test_cli_run.py's test_skera_budget_matches_hand_derived_numbers)."""
    hifi, sreads = test_bams
    mols = _annotated(hifi)
    out = tmp_path / "molecules.bam"
    n = molecules.write_molecules(hifi, mols, sreads, out)
    assert n == 30
    assert sum(1 for _ in pysam.AlignmentFile(str(out), "rb", check_sq=False)) == 30


def test_write_molecules_header_and_rg_match_the_template(test_bams, tmp_path):
    hifi, sreads = test_bams
    mols = _annotated(hifi)
    out = tmp_path / "molecules.bam"
    molecules.write_molecules(hifi, mols, sreads, out)
    with pysam.AlignmentFile(str(sreads), "rb", check_sq=False) as t:
        template_rg = t.header.to_dict()["RG"]
    with pysam.AlignmentFile(str(out), "rb", check_sq=False) as w:
        assert w.header.to_dict()["RG"] == template_rg
        for r in w:
            assert r.get_tag("RG") == template_rg[0]["ID"]


def test_write_molecules_sequences_and_qualities_equal_the_slices(test_bams, tmp_path):
    """The written (zm, qs, qe) set must be exactly the usable molecules of `mols`, and each record's
    sequence and qualities must equal the HiFi read sliced at that molecule's own start:end, taken
    from the molecules frame rather than the record's own qs/qe tags."""
    hifi, sreads = test_bams
    mols = _annotated(hifi)
    out = tmp_path / "molecules.bam"
    molecules.write_molecules(hifi, mols, sreads, out)

    expected = set(mols.filter(molecules.USABLE).select("zm", "start", "end").iter_rows())
    records = {(r.get_tag("zm"), r.get_tag("qs"), r.get_tag("qe")): r
               for r in pysam.AlignmentFile(str(out), "rb", check_sq=False)}
    assert set(records) == expected

    by_zm = {r.get_tag("zm"): (r.query_sequence, r.query_qualities) for r in pysam.AlignmentFile(str(hifi), "rb", check_sq=False)}
    for zm, start, end in expected:
        r = records[(zm, start, end)]
        seq, qual = by_zm[zm]
        assert r.query_sequence == seq[start:end]
        assert list(r.query_qualities) == list(qual[start:end])


def test_write_molecules_names_parse_as_movie_zm_ccs_start_end(test_bams, tmp_path):
    hifi, sreads = test_bams
    mols = _annotated(hifi)
    out = tmp_path / "molecules.bam"
    molecules.write_molecules(hifi, mols, sreads, out)
    for r in pysam.AlignmentFile(str(out), "rb", check_sq=False):
        movie, zm, ccs, se = r.query_name.split("/")
        assert movie == "movie"
        assert ccs == "ccs"
        s, e = (int(x) for x in se.split("_"))
        assert int(zm) == r.get_tag("zm")
        assert (s, e) == (r.get_tag("qs"), r.get_tag("qe"))
        assert r.flag == 4


def test_write_molecules_di_orders_molecules_of_a_read_by_start(test_bams, tmp_path):
    hifi, sreads = test_bams
    mols = _annotated(hifi)
    out = tmp_path / "molecules.bam"
    molecules.write_molecules(hifi, mols, sreads, out)
    by_zm = {}
    for r in pysam.AlignmentFile(str(out), "rb", check_sq=False):
        by_zm.setdefault(r.get_tag("zm"), []).append((r.get_tag("di"), r.get_tag("qs")))
    for rows in by_zm.values():
        rows.sort(key=lambda x: x[0])
        assert [di for di, _ in rows] == list(range(len(rows)))
        assert [qs for _, qs in rows] == sorted(qs for _, qs in rows)


def test_write_molecules_only_usable_end_classes_are_written(test_bams, tmp_path):
    """The junction molecules themselves (tso_tso at zm 4, 5; r1_r1 at zm 6) must not appear."""
    hifi, sreads = test_bams
    mols = _annotated(hifi)
    out = tmp_path / "molecules.bam"
    molecules.write_molecules(hifi, mols, sreads, out)
    written = {(r.get_tag("zm"), r.get_tag("qs"), r.get_tag("qe")) for r in pysam.AlignmentFile(str(out), "rb", check_sq=False)}
    not_usable = mols.filter((pl.col("end_class") != "proper") | pl.col("internal"))
    for zm, start, end in not_usable.select("zm", "start", "end").iter_rows():
        assert (zm, start, end) not in written


def test_write_molecules_keep_further_restricts_reads(test_bams, tmp_path):
    hifi, sreads = test_bams
    mols = _annotated(hifi)
    out = tmp_path / "molecules.bam"
    n = molecules.write_molecules(hifi, mols, sreads, out, keep=lambda zm: zm == 1)
    assert n == 8
    zms = {r.get_tag("zm") for r in pysam.AlignmentFile(str(out), "rb", check_sq=False)}
    assert zms == {1}


def test_write_molecules_records_follow_hifi_order_then_start_end(test_bams, tmp_path):
    """Records come in HiFi file order and, within a read, by (start, end), whatever the row order of the frame
    given: the same records, in the same order, as the dict of (start, end) lists write_molecules used to keep."""
    hifi, sreads = test_bams
    mols = _annotated(hifi).sample(fraction=1.0, shuffle=True, seed=3)
    out = tmp_path / "molecules.bam"
    molecules.write_molecules(hifi, mols, sreads, out)
    by_zm = {}
    for zm, start, end in mols.filter(molecules.USABLE).select("zm", "start", "end").iter_rows():
        by_zm.setdefault(zm, []).append((start, end))
    hifi_order = [r.get_tag("zm") for r in pysam.AlignmentFile(str(hifi), "rb", check_sq=False)]
    want = [(zm, di, s, e) for zm in hifi_order if zm in by_zm for di, (s, e) in enumerate(sorted(by_zm[zm]))]
    got = [(r.get_tag("zm"), r.get_tag("di"), r.get_tag("qs"), r.get_tag("qe")) for r in pysam.AlignmentFile(str(out), "rb", check_sq=False)]
    assert len(got) == 30
    assert got == want


@pytest.mark.parametrize(("batch", "fraction"), [(200, 1.0), (2, 1.0), (2, 0.5), (1, 0.000001)])
def test_streamed_molecules_bam_is_byte_identical_to_write_molecules(test_bams, tmp_path, monkeypatch, batch, fraction):
    """molecules.bam streamed by annotate() through a MoleculeSink holds the molecules of every read whatever the
    fraction, and has the bytes write_molecules gives over every read's molecules (one batch, batches of 2, a hash
    sample of half, a sample of zero reads)."""
    hifi, sreads = test_bams
    want = tmp_path / "write_molecules.bam"
    n = molecules.write_molecules(hifi, _annotated(hifi), sreads, want)
    monkeypatch.setattr(runner, "BATCH", batch)
    got = tmp_path / "streamed.bam"
    with molecules.MoleculeSink(sreads, got) as sink:
        runner.annotate(hifi, KINNEX8_SC3, fraction, 2, molecules_out=sink)
    assert (n, sink.written) == (30, 30)
    assert got.read_bytes() == want.read_bytes()


def test_sink_writes_molecules_bam_part_and_renames_it_when_the_block_ends(test_bams, tmp_path):
    """Until the with-block ends without an error the records go to molecules.bam.part; molecules.bam appears only
    then, complete."""
    hifi, sreads = test_bams
    mols = _annotated(hifi)
    out = tmp_path / "molecules.bam"
    with molecules.MoleculeSink(sreads, out) as sink:
        sink.add(list(pysam.AlignmentFile(str(hifi), "rb", check_sq=False)), mols)
        assert [p.name for p in tmp_path.iterdir()] == ["molecules.bam.part"]
    assert [p.name for p in tmp_path.iterdir()] == ["molecules.bam"]
    assert sum(1 for _ in pysam.AlignmentFile(str(out), "rb", check_sq=False)) == 30


@pytest.mark.parametrize("error", [BrokenProcessPool, KeyboardInterrupt])
@pytest.mark.parametrize("written", [True, False])
def test_sink_leaves_neither_file_when_the_block_raises(test_bams, tmp_path, error, written):
    """A run that fails (a killed worker: BrokenProcessPool) or is interrupted (Ctrl-C) inside the with-block leaves
    neither molecules.bam nor molecules.bam.part, whether or not records were written, never a truncated BAM that
    looks complete."""
    hifi, sreads = test_bams
    mols = _annotated(hifi)
    out = tmp_path / "molecules.bam"
    with pytest.raises(error):
        with molecules.MoleculeSink(sreads, out) as sink:
            if written:
                sink.add(list(pysam.AlignmentFile(str(hifi), "rb", check_sq=False)), mols)
                assert sink.written == 30
            raise error()
    assert list(tmp_path.iterdir()) == []


def test_write_molecules_does_not_hold_molecules_as_python_tuples(test_bams, tmp_path):
    """write_molecules finds a read's molecules by binary search in sorted columns, not in a dict of Python (start, end)
    tuples. 500,000 usable molecules of ZMWs the HiFi BAM does not hold (so nothing is written): as tuples they took a
    Python heap peak of 72 MB (measured before the change; under 0.1 MB after)."""
    hifi, sreads = test_bams
    i = np.arange(500_000)
    mols = pl.DataFrame({"zm": (1_000_000 + i // 10).astype(np.uint32), "start": (i % 10 * 700).astype(np.int32),
                         "end": (i % 10 * 700 + 650).astype(np.int32)}).with_columns(
        pl.lit("proper").alias("end_class"), pl.lit(False).alias("internal"))
    tracemalloc.start()
    try:
        n = molecules.write_molecules(hifi, mols, sreads, tmp_path / "molecules.bam")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert n == 0
    assert peak < 20 * 2**20


@pytest.mark.real
@pytest.mark.slow
def test_lima_pass_rate_on_calibration_mirror_molecules(local, tmp_path):
    """The brief's real check: molecules of 3,000 calibration mirror reads, run through
    lima --isoseq --peek-guess with the production primers, must pass at a rate of at least 0.998
    (v10's downstream test measured 0.9986 for the molecules skera itself does not write)."""
    hifi = local["calibration_dir"] / "nd_first200k.hifi.nomod.bam"
    template = local["calibration_dir"] / "nd_first200k.skera120.bam"
    env = str(local["pbtools_env"])

    read_summary, mols, _ = runner.annotate(hifi, KINNEX8_SC3, 1.0, 8)
    mirror_zm = sorted(read_summary.filter(pl.col("primary_pattern") == "mirror")["zm"].to_list())[:3000]
    chosen = set(mirror_zm)

    mol_bam = tmp_path / "molecules.bam"
    written = molecules.write_molecules(hifi, mols, template, mol_bam, keep=lambda zm: zm in chosen)

    isoseq_bam = tmp_path / "mol.isoseq.bam"
    subprocess.run(["conda", "run", "-n", env, "lima", "--isoseq", "--peek-guess", "-j", "8",
                    "--log-level", "WARN", str(mol_bam), str(PRIMERS), str(isoseq_bam)], check=True)

    passed_bam = tmp_path / "mol.isoseq.FlashBD_5p--FlashBD_3p.bam"
    passed = sum(1 for _ in pysam.AlignmentFile(str(passed_bam), "rb", check_sq=False))
    print(f"written={written} passed={passed} rate={passed / written}", file=sys.stderr)
    assert passed / written >= 0.998
