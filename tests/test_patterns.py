"""Tests for patterns module."""

import edlib

from kinnex_qc.patterns import (
    MAS,
    MAS_ORDER,
    TSO_CORE,
    TSO_EXT,
    R1,
    all_patterns,
    primer_patterns,
    revcomp,
)


def test_mas_sequences():
    """MAS dict contains all expected adapters."""
    assert len(MAS) == 9
    for letter in MAS_ORDER:
        assert letter in MAS
        if letter == "Q":
            assert len(MAS[letter]) == 14
        else:
            assert len(MAS[letter]) == 17


def test_mas_order():
    """MAS_ORDER has the correct sequence."""
    assert MAS_ORDER == ["A", "B", "C", "D", "E", "F", "G", "H", "Q"]


def test_primer_sequences_exist():
    """TSO and R1 core sequences are defined."""
    assert len(TSO_CORE) == 25
    assert len(TSO_EXT) == 5
    assert len(R1) == 20


def test_revcomp():
    """revcomp() returns correct reverse complement."""
    assert revcomp("AAAA") == "TTTT"
    assert revcomp("ACGT") == "ACGT"
    assert revcomp("GCTAGC") == "GCTAGC"
    assert revcomp("A") == "T"


def test_all_patterns():
    """all_patterns() returns 11 patterns in order."""
    patterns = all_patterns()
    assert len(patterns) == 11

    names = [p.name for p in patterns]
    expected = MAS_ORDER + ["TSO", "R1"]
    assert names == expected

    for p in patterns[:9]:
        assert p.kind == "MAS"
        assert p.max_edits == 2

    assert patterns[9].kind == "TSO"
    assert patterns[9].max_edits == 3
    assert patterns[10].kind == "R1"
    assert patterns[10].max_edits == 3


def test_primer_patterns():
    """primer_patterns() returns TSO and R1 only."""
    patterns = primer_patterns()
    assert len(patterns) == 2
    assert patterns[0].name == "TSO"
    assert patterns[1].name == "R1"


def test_minimum_edit_distance():
    """Minimum edit distance between all patterns is at least 5.

    The minimum edit distance between any two of these patterns,
    reverse complements included, is 5 (edlib HW mode, shorter pattern placed
    anywhere in the longer).
    """
    patterns = all_patterns()
    sequences = [(p.name, p.seq) for p in patterns]

    # Add reverse complements
    for p in patterns:
        sequences.append((f"{p.name}'", revcomp(p.seq)))

    n = len(sequences)
    min_dist = float("inf")
    min_pair = None

    for i in range(n):
        for j in range(i + 1, n):
            name_i, seq_i = sequences[i]
            name_j, seq_j = sequences[j]

            # Shorter in longer using HW mode
            if len(seq_i) <= len(seq_j):
                shorter, longer = seq_i, seq_j
            else:
                shorter, longer = seq_j, seq_i

            result = edlib.align(shorter, longer, task="HW")
            dist = result["editDistance"]

            if dist < min_dist:
                min_dist = dist
                min_pair = (name_i, name_j, dist)

    assert min_dist >= 5, f"Minimum edit distance is {min_dist}, found between {min_pair}"
