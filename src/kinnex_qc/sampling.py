import numpy as np

# splitmix64 constants (standard)
_SPLITMIX_B = 0xBF58476D1CE4E5B9
_SPLITMIX_C = 0x94D049BB133111EB
_MASK64 = (1 << 64) - 1
HASH_SAMPLE_SALT = 0x6B43A9B5   # salts the hash sample, so it does not line up with keep_zmw's 1-in-N subsamples


def splitmix64(x: int) -> int:
    """64-bit splitmix64 finaliser."""
    x = int(x)
    x = x & _MASK64
    x ^= x >> 30
    x = (x * _SPLITMIX_B) & _MASK64
    x ^= x >> 27
    x = (x * _SPLITMIX_C) & _MASK64
    x ^= x >> 31
    return x


def keep_zmw(zm: int, modulo: int) -> bool:
    """Check if ZMW should be kept based on subsampling modulo."""
    return splitmix64(int(zm)) % modulo == 0


def keep_zmw_array(zms, modulo: int) -> np.ndarray:
    """Vectorised keep_zmw over an array of ZMW ids (same hash, same answer)."""
    x = np.asarray(zms, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * np.uint64(_SPLITMIX_B)
        x = x ^ (x >> np.uint64(27))
        x = x * np.uint64(_SPLITMIX_C)
        x = x ^ (x >> np.uint64(31))
    return x % np.uint64(modulo) == 0


def in_hash_sample(zm: int, fraction: float) -> bool:
    """A ZMW is in the hash sample when its salted splitmix64 hash falls in the lowest fraction of the 64-bit range:
    the sample spreads over the whole cell, whatever the BAM order (a prefix of the HiFi BAM is not a random sample)."""
    return splitmix64(int(zm) ^ HASH_SAMPLE_SALT) < int(fraction * 2**64)
