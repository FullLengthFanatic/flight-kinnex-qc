"""Tests for classify.py."""
import random
import time

import pytest

from kinnex_qc.classify import END_CLASSES, classify_segment, tso_toward_lower
from kinnex_qc.patterns import MAS, R1, TSO_CORE, TSO_EXT, revcomp

rng = random.Random(7)


def rand(n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


CDNA = rand(1200)
CLS1, CLS2, CLS3, UMI = rand(9), rand(9), rand(9), rand(8)


def bead(shift: int = 0, polyt: int = 25, linker1: str = "GTGA") -> str:
    """R1 + optional shift bases + CLS1 GTGA CLS2 GACA CLS3 UMI + polyT."""
    return R1 + "C" * shift + CLS1 + linker1 + CLS2 + "GACA" + CLS3 + UMI + "T" * polyt


def layout_r1_first(cdna: str = CDNA, **kw) -> str:
    """R1 ... polyT cDNA(antisense) TSO' : R1 reads '+' in the head."""
    return bead(**kw) + cdna + revcomp(TSO_CORE + TSO_EXT)


def layout_tso_first(cdna: str = CDNA, **kw) -> str:
    """TSO ... cDNA polyA ... R1' : the same molecule read from the other strand."""
    return revcomp(layout_r1_first(cdna, **kw))


class TestEndClasses:
    def test_proper_r1_first(self):
        call = classify_segment(layout_r1_first())
        assert call.end_class == "proper"
        assert call.tso_side == "tail"
        assert call.internal is False
        assert call.bead_ok is True and call.bead_shift == 0
        assert call.polyT_anchored is True and call.polyT_len == 25
        # polyT start to TSO' core start: polyT + cDNA + revcomp(ATGGG)
        assert call.insert_len == 25 + len(CDNA) + len(TSO_EXT)

    def test_proper_tso_first_same_numbers(self):
        a = classify_segment(layout_r1_first())
        b = classify_segment(layout_tso_first())
        assert b.end_class == "proper" and b.tso_side == "head"
        assert (b.bead_ok, b.bead_shift, b.polyT_len, b.insert_len) == (a.bead_ok, a.bead_shift, a.polyT_len, a.insert_len)

    @pytest.mark.parametrize("shift", [0, 1, 2, 3])
    def test_bead_shift(self, shift):
        call = classify_segment(layout_r1_first(shift=shift))
        assert call.bead_ok and call.bead_shift == shift
        assert call.polyT_len == 25
        assert call.insert_len == 25 + len(CDNA) + len(TSO_EXT)

    def test_broken_linker_uses_fallback(self):
        call = classify_segment(layout_r1_first(linker1="CCCC"))
        assert call.end_class == "proper"
        assert call.bead_ok is False and call.polyT_anchored is False
        assert call.polyT_len == 25          # the fallback window reaches the tail
        assert call.insert_len == 43 + 25 + len(CDNA) + len(TSO_EXT)   # measured from the Read1 end

    @pytest.mark.parametrize("n", [0, 5, 8, 12, 20, 30])
    def test_polyt_lengths(self, n):
        # cDNA starting with G so the tail cannot run into it
        call = classify_segment(layout_r1_first(cdna="GGG" + CDNA, polyt=n))
        assert call.polyT_len == n

    def test_polyt_single_interruption_is_kept(self):
        seq = bead(polyt=0) + "T" * 10 + "A" + "T" * 10 + CDNA + revcomp(TSO_CORE)
        assert classify_segment(seq).polyT_len == 21

    def test_tso_tso(self):
        call = classify_segment(TSO_CORE + TSO_EXT + CDNA + revcomp(TSO_CORE + TSO_EXT))
        assert call.end_class == "tso_tso"
        assert call.insert_len is None and call.polyT_len is None

    def test_r1_r1_measures_polyt(self):
        call = classify_segment(bead(polyt=15) + CDNA + revcomp(bead(polyt=15)))
        assert call.end_class == "r1_r1"
        assert call.bead_ok and call.polyT_len == 15
        assert call.insert_len is None

    def test_tso_only(self):
        for seq in (TSO_CORE + CDNA, revcomp(TSO_CORE + CDNA)):
            call = classify_segment(seq)
            assert call.end_class == "tso_only" and call.tso_side is None

    def test_r1_only(self):
        call = classify_segment(bead(polyt=12) + CDNA)
        assert call.end_class == "r1_only" and call.polyT_len == 12
        call = classify_segment(revcomp(bead(polyt=12) + CDNA))
        assert call.end_class == "r1_only" and call.polyT_len == 12

    def test_none(self):
        call = classify_segment(CDNA)
        assert call.end_class == "none"
        assert call.head_hits == () and call.tail_hits == ()

    def test_wrong_strand_is_other(self):
        # TSO reverse complement at the start cannot come from a primer read from outside
        assert classify_segment(revcomp(TSO_CORE) + CDNA).end_class == "other"
        # wrong-strand primer at the far end of a single-sided segment
        assert classify_segment(TSO_CORE + CDNA + R1).end_class == "other"

    def test_short_tso_tso_counted_once(self):
        seq = TSO_CORE + rand(70) + revcomp(TSO_CORE)
        assert len(seq) == 120
        call = classify_segment(seq)
        assert call.end_class == "tso_tso"
        assert len(call.head_hits) == 1 and len(call.tail_hits) == 1
        assert call.internal is False

    def test_short_proper(self):
        call = classify_segment(layout_r1_first(cdna=rand(60), polyt=10))
        assert len(layout_r1_first(cdna=rand(60), polyt=10)) < 300
        assert call.end_class == "proper" and call.polyT_len == 10

    def test_all_classes_listed(self):
        assert set(END_CLASSES) == {"proper", "tso_tso", "r1_r1", "tso_only", "r1_only", "none", "other"}


class TestInternal:
    def test_fused_without_adapter(self):
        # two proper molecules joined with the MAS adapter lost
        seq = layout_tso_first() + layout_r1_first()
        call = classify_segment(seq)
        assert call.internal is True
        assert {h.kind for h in call.internal_hits} >= {"R1"}

    def test_fused_with_missed_adapter(self):
        seq = layout_tso_first() + MAS["C"] + layout_r1_first()
        call = classify_segment(seq)
        assert call.internal is True
        assert any(h.name == "C" for h in call.internal_hits)

    def test_interior_mas_alone_is_listed_not_internal(self):
        cdna = CDNA[:600] + MAS["F"] + CDNA[600:]
        call = classify_segment(layout_r1_first(cdna=cdna))
        assert call.end_class == "proper" and call.internal is False
        assert [h.name for h in call.internal_hits] == ["F"]


class TestTsoTowardLower:
    def test_directions(self):
        head = classify_segment(layout_tso_first())   # TSO in head
        tail = classify_segment(layout_r1_first())    # TSO in tail
        assert tso_toward_lower(head, 3, 4) is True
        assert tso_toward_lower(head, 4, 3) is False
        assert tso_toward_lower(tail, 4, 3) is True
        assert tso_toward_lower(tail, 3, 4) is False
        assert tso_toward_lower(classify_segment(CDNA), 3, 4) is None


@pytest.mark.real
class TestMiniData:
    def test_nd_and_kin(self, mini):
        from pipeline_helpers import bam_records, mini_samples
        samples = mini_samples(mini)
        stats = {}
        for sample in ("BD144_ND", "BD144_kin"):
            recs = bam_records(samples[sample].segmented_bam)
            t0 = time.perf_counter()
            calls = [classify_segment(r.query_sequence) for r in recs]
            us = (time.perf_counter() - t0) / len(recs) * 1e6
            n = len(calls)
            stats[sample] = {c: sum(x.end_class == c for x in calls) / n for c in END_CLASSES}
            print(f"{sample}: {n} S-reads, {us:.0f} us/segment, {stats[sample]}")
            assert stats[sample]["proper"] > 0.85
            # a non-positive insert only happens for empty molecules (TSO on a truncated bead oligo)
            assert all(c.insert_len > 0 for r, c in zip(recs, calls) if c.end_class == "proper" and len(r.query_sequence) >= 150)
        assert stats["BD144_kin"]["tso_tso"] < stats["BD144_ND"]["tso_tso"]
