"""Tests for `--write-molecules`: a BAM of the molecules a full-read segmentation finds."""
import subprocess
import sys
from pathlib import Path

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
