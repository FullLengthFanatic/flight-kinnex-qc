"""Tests for annotate.py."""
import collections
import random

import pytest

from kinnex_qc.annotate import adapter_chain, annotate_read, chain_flags, primary_pattern, segment_hits, split_segments
from kinnex_qc.classify import call_from_hits
from kinnex_qc.patterns import MAS, MAS_ORDER, R1, TSO_CORE, TSO_EXT, revcomp
from kinnex_qc.scan import Hit

rng = random.Random(11)


def rand(n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def molecule(n):
    """One cDNA molecule as it sits in an array: R1 + bead + polyT + cDNA + TSO'."""
    return R1 + rand(43) + "T" * 20 + rand(n) + revcomp(TSO_CORE + TSO_EXT)


def mutate(seq, rate):
    out = []
    for c in seq:
        r = rng.random()
        if r < rate / 3:
            out.append(rng.choice("ACGT".replace(c, "")))
        elif r < 2 * rate / 3:
            continue
        elif r < rate:
            out.append(c + rng.choice("ACGT"))
        else:
            out.append(c)
    return "".join(out)


def array(order=MAS_ORDER, mols=None):
    """Adapters in the given order with a molecule between each pair; returns (seq, adapter starts)."""
    mols = mols or [molecule(rng.randint(300, 3000)) for _ in range(len(order) - 1)]
    seq, starts = "", []
    for i, a in enumerate(order):
        name, minus = a.rstrip("'"), a.endswith("'")
        starts.append(len(seq))
        seq += revcomp(MAS[name]) if minus else MAS[name]
        if i < len(order) - 1:
            seq += mols[i]
    return seq, starts


def names(hits):
    return [(h.name if h.strand == "+" else h.name + "'") for h in hits if h.kind == "MAS"]


class TestAnnotateRead:
    @pytest.mark.parametrize("rc", [False, True])
    def test_full_array_both_orientations(self, rc):
        seq, starts = array()
        if rc:
            seq = revcomp(seq)
        hits = annotate_read(mutate(seq, 0.01))
        tokens, flipped = adapter_chain(hits)
        assert tokens == MAS_ORDER
        assert flipped is rc

    def test_positions_exact_without_errors(self):
        seq, starts = array()
        mas = [h for h in annotate_read(seq) if h.kind == "MAS"]
        assert [h.start for h in mas] == starts
        assert all(h.edits == 0 for h in mas)

    def test_primers_found_between_adapters(self):
        seq, _ = array()
        kinds = collections.Counter(h.kind for h in annotate_read(seq))
        assert kinds["TSO"] == 8 and kinds["R1"] == 8 and kinds["MAS"] == 9

    def test_adapter_inside_cdna_is_dropped(self):
        mols = [molecule(800) for _ in range(8)]
        mols[3] = R1 + rand(43) + "T" * 20 + rand(400) + MAS["D"] + rand(400) + revcomp(TSO_CORE)
        seq, _ = array(mols=mols)
        assert names(annotate_read(seq)) == MAS_ORDER


class TestChains:
    def test_complete(self):
        assert chain_flags(MAS_ORDER) == {"complete"}
        assert primary_pattern(chain_flags(MAS_ORDER)) == "complete"

    def test_truncated(self):
        assert chain_flags(["B", "C", "D", "E", "F", "G", "H", "Q"]) == {"truncated"}
        assert chain_flags(["A", "B", "C", "D", "E", "F", "G", "H"]) == {"truncated"}

    def test_mirror_from_array(self):
        order = ["A", "B", "C", "D", "D'", "C'", "B'", "A'"]
        seq, _ = array(order)
        tokens, _ = adapter_chain(annotate_read(seq))
        assert "mirror" in chain_flags(tokens)
        assert primary_pattern(chain_flags(tokens)) == "mirror"

    def test_skipped_slot(self):
        tokens = ["A", "B", "C", "E", "F", "G", "H", "Q"]
        assert chain_flags(tokens) == {"skipped_slot"}

    def test_skipped_slot_on_minus_run(self):
        """A "-" run reads downwards (Q' H' G' ...), so a skip there is a drop of more than one letter."""
        assert chain_flags("Q' H' F' E' E F G H Q".split()) == {"mirror", "skipped_slot"}      # G' missing
        assert primary_pattern(chain_flags("Q' H' D' E F G H Q".split())) == "skipped_slot"
        assert "skipped_slot" not in chain_flags("A B C D D' C' B' A'".split())
        assert chain_flags(["B'", "D'", "E", "F", "G"]) == {"unresolved"}                  # out of order, not a skip

    def test_repeated_not_adjacent(self):
        assert "repeated" in chain_flags(["A", "B", "C", "A", "B"])

    def test_priority(self):
        assert primary_pattern({"truncated", "mirror"}) == "mirror"
        assert primary_pattern({"skipped_slot", "repeated"}) == "skipped_slot"

    def test_unresolved(self):
        assert chain_flags([]) == {"unresolved"}
        assert chain_flags(["C", "B"]) == {"unresolved"}

    def test_normalisation(self):
        hits = [Hit(n, "MAS", "-", i * 100, i * 100 + 17, 0) for i, n in enumerate(["Q", "H", "G"])]
        tokens, flipped = adapter_chain(hits)
        assert flipped and tokens == ["G", "H", "Q"]
        mixed = [Hit("H", "MAS", "-", 0, 17, 0), Hit("H", "MAS", "+", 500, 517, 0), Hit("Q", "MAS", "+", 900, 914, 0)]
        assert adapter_chain(mixed) == (["H'", "H", "Q"], False)


class TestSplitSegments:
    def test_tokens_and_edges(self):
        hits = [Hit("A", "MAS", "+", 60, 77, 0), Hit("B", "MAS", "-", 500, 517, 1)]
        assert split_segments(900, hits) == [(0, 60, None, "A"), (77, 500, "A", "B'"), (517, 900, "B'", None)]

    def test_short_edges_dropped(self):
        hits = [Hit("A", "MAS", "+", 10, 27, 0), Hit("B", "MAS", "+", 500, 517, 0)]
        assert split_segments(540, hits) == [(27, 500, "A", "B")]

    def test_no_adapter(self):
        assert split_segments(40, []) == []
        assert split_segments(400, []) == [(0, 400, None, None)]


def bead_molecule(n, r):
    """R1, bead (GTGA 29 nt and GACA 42 nt after the R1 start, CLS3 and UMI), 20 nt polyT, n nt cDNA, TSO'."""
    seq = lambda k: "".join(r.choice("ACGT") for _ in range(k))
    return R1 + seq(9) + "GTGA" + seq(9) + "GACA" + seq(17) + "T" * 20 + seq(n) + revcomp(TSO_CORE + TSO_EXT)


class TestSegmentHits:
    def test_primer_ending_inside_next_adapter(self):
        """mini BD144_kin zm 196412973: TSO' [813, 838) with 0 edits, then B [837, 854) with 1 edit."""
        hits = [Hit("A", "MAS", "+", 1, 18, 0), Hit("R1", "R1", "+", 20, 40, 0),
                Hit("TSO", "TSO", "-", 813, 838, 0), Hit("B", "MAS", "+", 837, 854, 1)]
        head, tail, interior = segment_hits(hits, 18, 837)
        assert [(h.name, h.start, h.end) for h in head] == [("R1", 2, 22)]
        assert [(h.name, h.start, h.end) for h in tail] == [("TSO", 795, 820)]     # not clipped to the length 819
        assert interior == []

    def test_primer_starting_inside_previous_adapter(self):
        hits = [Hit("D", "MAS", "+", 100, 117, 0), Hit("R1", "R1", "+", 115, 136, 1)]
        assert [(h.start, h.end) for h in segment_hits(hits, 117, 900)[0]] == [(-2, 19)]

    def test_assignment_by_midpoint(self):
        tso = Hit("TSO", "TSO", "-", 880, 905, 0)                                  # midpoint 892.5
        assert segment_hits([tso], 100, 892) == ([], [], [])
        assert [h.name for h in segment_hits([tso], 100, 893)[1]] == ["TSO"]
        mas = Hit("C", "MAS", "+", 890, 907, 0)                                    # a MAS hit must lie fully inside
        assert segment_hits([mas], 100, 900) == ([], [], [])

    def test_primers_overlapping_adapters_keep_proper_segments(self):
        """B lost its first base, so B aligns 1 bp into the TSO' before it; the Read1 after D lost its first base,
        so it aligns into D. All eight molecules stay proper, bead_ok, with the right insert length."""
        r = random.Random(5)
        ns = [r.randint(400, 900) for _ in range(8)]
        mols = [bead_molecule(n, r) for n in ns]
        mols[3] = mols[3][1:]
        seq = "".join((MAS[a][1:] if a == "B" else MAS[a]) + (mols[i] if i < 8 else "") for i, a in enumerate(MAS_ORDER))
        hits = annotate_read(seq)
        mas = {h.name: h for h in hits if h.kind == "MAS"}
        primers = [h for h in hits if h.kind != "MAS"]
        assert any(h.name == "TSO" and h.end > mas["B"].start for h in primers)
        assert any(h.name == "R1" and mas["D"].start < h.start < mas["D"].end for h in primers)
        calls = [call_from_hits(seq[a:b], *segment_hits(hits, a, b)) for a, b, _, _ in split_segments(len(seq), hits)]
        assert [c.end_class for c in calls] == ["proper"] * 8
        assert all(c.bead_ok for c in calls)
        assert [c.insert_len for c in calls] == [20 + n + len(TSO_EXT) for n in ns]
