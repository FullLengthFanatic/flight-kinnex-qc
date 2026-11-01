"""Tests for scan.py."""
import random

from kinnex_qc.patterns import MAS, R1, TSO_CORE, Pattern, revcomp
from kinnex_qc.scan import (
    Hit, windows, best_hit, find_all, scan_segment, encode_hits, decode_hits,
    bead_check, BeadCall, polyt_length, polyt_fallback, LINKER1, LINKER2,
    LINKER1_OFFSET, LINKER2_AFTER_LINKER1
)


class TestWindows:
    """Test window boundary calculations."""

    def test_long_segment(self):
        """Segment >= 300 bp uses fixed 150 bp windows."""
        head, tail, interior = windows(300)
        assert head == (0, 150)
        assert tail == (150, 300)
        assert interior == (150, 150)  # Empty interior

    def test_long_segment_with_interior(self):
        """Large segment has meaningful interior."""
        head, tail, interior = windows(500)
        assert head == (0, 150)
        assert tail == (350, 500)
        assert interior == (150, 350)

    def test_short_segment(self):
        """Segment < 300 bp splits at midpoint."""
        head, tail, interior = windows(200)
        assert head == (0, 100)
        assert tail == (100, 200)
        assert interior is None

    def test_very_short_segment(self):
        """Very short segment."""
        head, tail, interior = windows(50)
        assert head == (0, 25)
        assert tail == (25, 50)
        assert interior is None


class TestBestHit:
    """Test best_hit function."""

    def test_exact_match_forward(self):
        """Exact match on forward strand."""
        seq = "AAAA" + LINKER1 + "GGGG"
        pattern = Pattern(name="test", kind="test", seq=LINKER1, max_edits=2)
        hit = best_hit(seq, pattern, "+", 0, len(seq))
        assert hit is not None
        assert hit.start == 4
        assert hit.end == 8
        assert hit.edits == 0
        assert hit.strand == "+"

    def test_exact_match_reverse(self):
        """Exact match on reverse strand."""
        seq = "AAAA" + revcomp(LINKER1) + "GGGG"
        pattern = Pattern(name="test", kind="test", seq=LINKER1, max_edits=2)
        hit = best_hit(seq, pattern, "-", 0, len(seq))
        assert hit is not None
        assert hit.strand == "-"
        assert hit.edits == 0

    def test_within_threshold(self):
        """Match within edit threshold."""
        seq = "AAAA" + "GTXA" + "GGGG"  # One mismatch in GTGA
        pattern = Pattern(name="test", kind="test", seq=LINKER1, max_edits=2)
        hit = best_hit(seq, pattern, "+", 0, len(seq))
        assert hit is not None
        assert hit.edits <= pattern.max_edits

    def test_exceeds_threshold(self):
        """Match exceeds edit threshold."""
        seq = "AAAA" + "XXXX" + "GGGG"  # Many mismatches
        pattern = Pattern(name="test", kind="test", seq=LINKER1, max_edits=0)
        hit = best_hit(seq, pattern, "+", 0, len(seq))
        assert hit is None

    def test_window_boundary(self):
        """Only searches within window."""
        seq = LINKER1 + "AAAA" + LINKER1
        pattern = Pattern(name="test", kind="test", seq=LINKER1, max_edits=0)
        hit = best_hit(seq, pattern, "+", 4, 8)  # Window excludes both hits
        assert hit is None


class TestFindAll:
    """Test find_all function (repeated hits with masking)."""

    def test_single_hit(self):
        """Single hit found."""
        seq = "AAA" + LINKER1 + "GGG"
        pattern = Pattern(name="test", kind="test", seq=LINKER1, max_edits=0)
        hits = find_all(seq, pattern, "+", 0, len(seq))
        assert len(hits) == 1

    def test_multiple_hits_masked(self):
        """Multiple non-overlapping hits found."""
        seq = LINKER1 + "AAAA" + LINKER1
        pattern = Pattern(name="test", kind="test", seq=LINKER1, max_edits=0)
        hits = find_all(seq, pattern, "+", 0, len(seq))
        assert len(hits) == 2


class TestEncodeDecodeHits:
    """Test hit encoding and decoding."""

    def test_empty(self):
        """Empty hit list."""
        s = encode_hits([])
        assert s == ""
        assert decode_hits("") == []

    def test_round_trip(self):
        """Encode then decode."""
        hits = [
            Hit(name="A", kind="MAS", strand="+", start=10, end=27, edits=1),
            Hit(name="TSO", kind="TSO", strand="-", start=50, end=75, edits=2),
        ]
        encoded = encode_hits(hits)
        decoded = decode_hits(encoded)
        assert len(decoded) == 2
        assert decoded[0].name == "A"
        assert decoded[0].start == 10
        assert decoded[1].edits == 2


class TestBeadCheck:
    """Test bead_check function."""

    def test_perfect_match_no_shift(self):
        """Perfect match with shift=0."""
        # Build oseq with perfect linker match
        r1_start = 10
        linker2_pos = r1_start + LINKER1_OFFSET + LINKER2_AFTER_LINKER1
        polyt_pos = linker2_pos + len(LINKER2) + 9 + 8

        oseq = (
            "A" * r1_start +  # Prefix
            "A" * LINKER1_OFFSET +  # Gap after R1
            LINKER1 +  # Perfect LINKER1
            "A" * (LINKER2_AFTER_LINKER1 - len(LINKER1)) +  # Gap
            LINKER2 +  # Perfect LINKER2
            "A" * (9 + 8)  # CLS3 + UMI
        )

        call = bead_check(oseq, r1_start)
        assert call.ok is True
        assert call.shift == 0
        assert call.polyT_anchor == polyt_pos

    def test_shift_2(self):
        """Match with shift=2."""
        r1_start = 10

        oseq = (
            "A" * r1_start +
            "A" * (LINKER1_OFFSET + 2) +
            LINKER1 +
            "A" * (LINKER2_AFTER_LINKER1 - len(LINKER1)) +
            LINKER2 +
            "A" * (9 + 8)
        )

        call = bead_check(oseq, r1_start)
        assert call.ok is True
        assert call.shift == 2

    def test_linker1_mismatch(self):
        """LINKER1 with one mismatch allowed."""
        r1_start = 10

        # LINKER1 = "GTGA", change one base to "GTXA"
        linker1_mut = "GTXA"
        oseq = (
            "A" * r1_start +
            "A" * LINKER1_OFFSET +
            linker1_mut +
            "A" * (LINKER2_AFTER_LINKER1 - len(linker1_mut)) +
            LINKER2 +
            "A" * (9 + 8)
        )

        call = bead_check(oseq, r1_start)
        assert call.ok is True

    def test_linker1_too_many_mismatches(self):
        """LINKER1 with too many mismatches."""
        r1_start = 10
        linker1_mut = "XXXX"  # All mismatches

        oseq = (
            "A" * r1_start +
            "A" * LINKER1_OFFSET +
            linker1_mut +
            "A" * (LINKER2_AFTER_LINKER1 - len(linker1_mut)) +
            LINKER2 +
            "A" * (9 + 8)
        )

        call = bead_check(oseq, r1_start)
        assert call.ok is False

    def test_linker2_mismatch(self):
        """LINKER2 with one mismatch allowed."""
        r1_start = 10

        linker2_mut = "GXCA"  # One mismatch in GACA
        oseq = (
            "A" * r1_start +
            "A" * LINKER1_OFFSET +
            LINKER1 +
            "A" * (LINKER2_AFTER_LINKER1 - len(LINKER1)) +
            linker2_mut +
            "A" * (9 + 8)
        )

        call = bead_check(oseq, r1_start)
        assert call.ok is True

    def test_out_of_bounds(self):
        """Linker extends past end of sequence."""
        r1_start = 10
        oseq = "A" * 40  # Short sequence

        call = bead_check(oseq, r1_start)
        assert call.ok is False
        assert call.shift is None
        assert call.polyT_anchor is None


class TestPolyTLength:
    """Test polyT length measurement."""

    def test_no_polyt(self):
        """No T at start."""
        seq = "AAAA"
        length = polyt_length(seq, 0)
        assert length == 0

    def test_pure_polyt(self):
        """Pure T stretch."""
        seq = "TTTTTT"
        length = polyt_length(seq, 0)
        assert length == 6

    def test_polyt_with_slack(self):
        """T starting within slack."""
        seq = "AATTTT"
        length = polyt_length(seq, 0)  # First T at position 1 (slack=2)
        assert length == 4

    def test_polyt_interruption_within_window(self):
        """Single non-T allowed per POLYT_WINDOW."""
        # POLYT_WINDOW = 8, POLYT_MAX_NONT = 1
        seq = "TTTTTXTT"  # Non-T at position 5
        length = polyt_length(seq, 0)
        # Should extend as long as every 8-window has <= 1 non-T
        assert length > 0

    def test_polyt_breaks_on_too_many_nont(self):
        """Multiple non-T in a window breaks stretch."""
        # Create sequence where window has 2+ non-T
        seq = "TTTXXXXTT"  # Multiple non-T
        length = polyt_length(seq, 0)
        # Should stop when window has too many non-T
        assert length < len(seq)

    def test_polyt_trim_trailing_nont(self):
        """Trailing non-T trimmed."""
        seq = "TTTTAX"
        length = polyt_length(seq, 0)
        # Should be 4 (the four Ts), trimming the "A" and "X"
        assert length == 4


class TestPolyTFallback:
    """Test polyT fallback measurement."""

    def test_fallback_first_t(self):
        """Find T-dominant stretch starting soon after position."""
        r1_end = 10
        seq = "A" * 10 + "AATTTTT"  # T-stretch starts at position 12
        length = polyt_fallback(seq, r1_end)
        assert length > 0

    def test_fallback_no_t(self):
        """No T in fallback search range."""
        r1_end = 10
        seq = "A" * 60  # No T
        length = polyt_fallback(seq, r1_end)
        assert length == 0


class TestScanSegment:
    """Test full segment scanning."""

    def test_scan_with_defaults(self):
        """TSO forward in the head, Read1 reverse complemented in the tail, interior empty."""
        rng = random.Random(0)
        filler = "".join(rng.choice("ACGT") for _ in range(400 - len(TSO_CORE) - len(R1)))
        seq = TSO_CORE + filler + revcomp(R1)
        assert len(seq) == 400
        head_hits, tail_hits, interior_hits = scan_segment(seq)
        assert [(h.name, h.strand) for h in head_hits] == [("TSO", "+")]
        assert [(h.name, h.strand) for h in tail_hits] == [("R1", "-")]
        assert interior_hits == []

    def test_scan_sorted(self):
        """Hits are sorted by start position."""
        seq = "AAA" + LINKER2 + "GGG" + LINKER1 + "TTT"
        head_hits, tail_hits, interior_hits = scan_segment(seq)

        # All hits should be sorted by start
        all_hits = head_hits + tail_hits + interior_hits
        for i in range(len(all_hits) - 1):
            assert all_hits[i].start <= all_hits[i+1].start


class TestScanWindowEdges:
    """A MAS hit straddling a window edge is reported once, in the window holding its midpoint.

    400 bp segment: head [0, 150), interior [150, 250), tail [250, 400).
    """

    FILLER = "".join(random.Random(0).choice("ACGT") for _ in range(400))

    def _seq_with_mas_a_at(self, start):
        mas_a = MAS["A"]
        end = start + len(mas_a)
        return self.FILLER[:start] + mas_a + self.FILLER[end:]

    def test_filler_has_no_hits(self):
        """The plain filler holds no MAS, TSO or R1 hit, so the edge tests are not flaky."""
        head_hits, tail_hits, interior_hits = scan_segment(self.FILLER)
        assert head_hits == []
        assert tail_hits == []
        assert interior_hits == []

    def test_hit_at_interior_midpoint(self):
        """MAS A at [145, 162), midpoint 153.5, is reported once, in the interior."""
        seq = self._seq_with_mas_a_at(145)
        head_hits, tail_hits, interior_hits = scan_segment(seq)
        assert head_hits == []
        assert tail_hits == []
        assert [(h.name, h.start, h.end) for h in interior_hits] == [("A", 145, 162)]

    def test_hit_at_head_midpoint(self):
        """MAS A at [140, 157), midpoint 148.5, is reported once, in the head."""
        seq = self._seq_with_mas_a_at(140)
        head_hits, tail_hits, interior_hits = scan_segment(seq)
        assert tail_hits == []
        assert interior_hits == []
        assert [(h.name, h.start, h.end) for h in head_hits] == [("A", 140, 157)]

    def test_hit_at_tail_midpoint(self):
        """MAS A at [243, 260), midpoint 251.5, is reported once, in the tail."""
        seq = self._seq_with_mas_a_at(243)
        head_hits, tail_hits, interior_hits = scan_segment(seq)
        assert head_hits == []
        assert interior_hits == []
        assert [(h.name, h.start, h.end) for h in tail_hits] == [("A", 243, 260)]
