"""Tests for the determinism of tests/make_test_bam.py's fixture builder, and its command line."""
import subprocess
import sys

import pysam

import make_test_bam
import synth


def _hifi_lengths(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return [len(r.query_sequence) for r in bam]


def test_build_independent_of_prior_rng_draws(tmp_path):
    """build() must not depend on how many draws other tests made on the shared synth.rng first."""
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()

    synth.rng.seed(11)
    hifi_a, _ = make_test_bam.build(a_dir)
    lens_a = _hifi_lengths(hifi_a)

    synth.rng.seed(11)
    synth.rng.random()          # simulate a draw made by some earlier test in the session
    hifi_b, _ = make_test_bam.build(b_dir)
    lens_b = _hifi_lengths(hifi_b)

    assert lens_a == lens_b


def test_build_does_not_disturb_shared_rng_state(tmp_path):
    """build() must leave the shared synth.rng exactly where it found it, so later draws (e.g.
    test_annotate's) are unaffected by whether the fixture was built first."""
    c_dir = tmp_path / "c"
    c_dir.mkdir()

    synth.rng.seed(11)
    before = synth.rng.getstate()
    make_test_bam.build(c_dir)
    after_draw = synth.rng.random()

    synth.rng.setstate(before)
    expected_draw = synth.rng.random()

    assert after_draw == expected_draw


def _records(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return [(r.query_name, r.query_sequence, r.get_tags(with_value_type=True)) for r in bam]


def test_command_line_writes_the_bams_build_writes(tmp_path):
    """`python tests/make_test_bam.py OUT_DIR`, as the README runs it, makes OUT_DIR and writes the same reads as build()."""
    out = tmp_path / "qc" / "test_bam"
    r = subprocess.run([sys.executable, make_test_bam.__file__, str(out)], capture_output=True, text=True, check=True)
    assert r.stdout.split() == [str(out / "hifi.bam"), str(out / "sreads.bam")]
    ref = tmp_path / "ref"
    ref.mkdir()
    for a, b in zip(make_test_bam.build(ref), (out / "hifi.bam", out / "sreads.bam"), strict=True):
        assert _records(b) == _records(a)
