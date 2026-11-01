"""Tests for the determinism of tests/make_test_bam.py's fixture builder."""
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
