import numpy as np

from kinnex_qc.sampling import splitmix64, keep_zmw, keep_zmw_array


class TestSplitmix64:
    """Test the splitmix64 hash function with known values."""

    def test_splitmix64_known_values(self):
        # Consistent output test vectors
        assert splitmix64(0) == 0x0
        assert splitmix64(1) == 0x5692161d100b05e5
        assert splitmix64(42) == 0xa759ea27d4727622

    def test_splitmix64_deterministic(self):
        # Same input must give same output
        for x in [0, 1, 42, 1000, 2**32, 2**63-1]:
            result1 = splitmix64(x)
            result2 = splitmix64(x)
            assert result1 == result2
            assert 0 <= result1 < 2**64


class TestKeepZmw:
    """Test ZMW subsampling."""

    def test_keep_zmw_rate(self):
        # On 100k random ZMW IDs, the rate should be close to 1/modulo
        zmws = np.arange(1000000)

        for modulo in [10, 100]:
            kept = sum(1 for zm in zmws if keep_zmw(zm, modulo))
            rate = kept / len(zmws)
            expected_rate = 1.0 / modulo
            # Allow 2% deviation
            assert abs(rate - expected_rate) < expected_rate * 0.02, \
                f"Rate {rate} far from expected {expected_rate}"


def test_in_hash_sample_fraction_and_independence():
    """The hash sample keeps about the requested fraction, and its 1-in-10 keep_zmw subsample stays about 1 in 10
    (the salt keeps the two hashes from lining up)."""
    from kinnex_qc.sampling import in_hash_sample
    zms = np.arange(1_000_000, 1_200_000)
    keep = np.array([in_hash_sample(z, 0.18) for z in zms])
    assert abs(keep.mean() - 0.18) < 0.005
    assert in_hash_sample(123456, 0.18) == in_hash_sample(123456, 0.18)
    sub = keep_zmw_array(zms[keep], 10)
    assert abs(sub.mean() - 0.1) < 0.006
    assert not any(in_hash_sample(z, 1e-12) for z in zms[:1000])


def test_hash_key_orders_the_hash_sample():
    """in_hash_sample(zm, f) is hash_key(zm) below f of the 64-bit range, so every hash sample is a prefix of the
    ZMWs sorted by hash_key: the order report.html picks its example reads in."""
    from kinnex_qc.sampling import hash_key, in_hash_sample
    zms = range(1_000_000, 1_020_000)
    ordered = sorted(zms, key=hash_key)
    for f in (0.01, 0.18, 0.5):
        assert all(in_hash_sample(z, f) == (hash_key(z) < int(f * 2**64)) for z in zms)
        sample = [z for z in ordered if in_hash_sample(z, f)]
        assert sample == ordered[:len(sample)]


def test_vectorised_hashes_equal_the_scalar_ones():
    """splitmix64_array, keep_zmw_array and hash_key_array give, value for value, what splitmix64, keep_zmw and
    hash_key give (report.html orders its example reads with hash_key_array)."""
    from kinnex_qc.sampling import hash_key, hash_key_array, splitmix64_array
    rng = np.random.default_rng(3)
    zms = np.concatenate([np.array([0, 1, 42, 2**31 - 1, 2**32 - 1], dtype=np.uint64),
                          np.arange(1_000_000, 1_005_000, dtype=np.uint64), rng.integers(0, 2**32, 5000, dtype=np.uint64)])
    assert splitmix64_array(zms).tolist() == [splitmix64(z) for z in zms.tolist()]
    assert keep_zmw_array(zms, 10).tolist() == [keep_zmw(z, 10) for z in zms.tolist()]
    assert hash_key_array(zms).tolist() == [hash_key(z) for z in zms.tolist()]
    assert hash_key_array(zms.astype(np.uint32)).tolist() == [hash_key(z) for z in zms.tolist()]
