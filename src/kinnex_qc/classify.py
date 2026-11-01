"""Segment end classification."""
from dataclasses import dataclass
from typing import Optional

from .patterns import revcomp
from .scan import Hit, scan_segment, bead_check, polyt_length, polyt_fallback


END_CLASSES = ("proper", "tso_tso", "r1_r1", "tso_only", "r1_only", "none", "other")


@dataclass(frozen=True)
class SegmentCall:
    end_class: str
    tso_side: Optional[str]          # "head" | "tail" for proper segments, else None
    head_hits: tuple[Hit, ...]
    tail_hits: tuple[Hit, ...]
    internal_hits: tuple[Hit, ...]
    internal: bool                   # an interior TSO or R1 hit (fusion evidence)
    bead_ok: bool
    bead_shift: Optional[int]
    polyT_len: Optional[int]         # None when no R1 end
    polyT_anchored: Optional[bool]
    insert_len: Optional[int]        # proper only


def _end_primers(hits: list[Hit], valid_strand: str) -> tuple[set[str], set[str]]:
    """(primers on the valid strand, primers on the wrong strand) among TSO/R1 hits of one end window."""
    valid = {h.name for h in hits if h.kind in ("TSO", "R1") and h.strand == valid_strand}
    wrong = {h.name for h in hits if h.kind in ("TSO", "R1") and h.strand != valid_strand}
    return valid, wrong


def _first(hits: list[Hit], name: str, strand: str) -> Hit:
    return min((h for h in hits if h.name == name and h.strand == strand), key=lambda h: h.edits)


def classify_segment(seq: str) -> SegmentCall:
    """Classify one segment sequence: scan it (scan_segment), then call_from_hits."""
    return call_from_hits(seq, *scan_segment(seq))


def call_from_hits(seq: str, head_hits: list[Hit], tail_hits: list[Hit], interior_hits: list[Hit]) -> SegmentCall:
    """Classify a segment by the primers in its end windows, given its hits in segment coordinates.

    A primer read from the outside in is "+" in the head window and "-" in the tail window.
    internal_hits lists interior hits of every pattern. Only an interior TSO or R1 hit makes the
    segment internal: every molecule ends in TSO or R1, so two fused molecules always show one,
    while a MAS match alone is often cDNA sequence (Q is 14 nt, D is low complexity).
    """
    head_valid, head_wrong = _end_primers(head_hits, "+")
    tail_valid, tail_wrong = _end_primers(tail_hits, "-")

    tso_side = None
    if head_wrong or tail_wrong or len(head_valid) > 1 or len(tail_valid) > 1:
        end_class = "other"
    elif head_valid == {"TSO"} and tail_valid == {"R1"}:
        end_class, tso_side = "proper", "head"
    elif head_valid == {"R1"} and tail_valid == {"TSO"}:
        end_class, tso_side = "proper", "tail"
    elif head_valid == {"TSO"} and tail_valid == {"TSO"}:
        end_class = "tso_tso"
    elif head_valid == {"R1"} and tail_valid == {"R1"}:
        end_class = "r1_r1"
    elif (head_valid, tail_valid) in (({"TSO"}, set()), (set(), {"TSO"})):
        end_class = "tso_only"
    elif (head_valid, tail_valid) in (({"R1"}, set()), (set(), {"R1"})):
        end_class = "r1_only"
    else:
        end_class = "none"

    # Orient so that an R1 end reads "+" at the start of oseq.
    r1_hit = None
    if "R1" in head_valid:
        r1_hit = _first(head_hits, "R1", "+")
        oseq, r1_start, r1_end = seq, r1_hit.start, r1_hit.end
    elif "R1" in tail_valid:
        r1_hit = _first(tail_hits, "R1", "-")
        L = len(seq)
        oseq, r1_start, r1_end = revcomp(seq), L - r1_hit.end, L - r1_hit.start

    bead_ok, bead_shift, polyT_len, polyT_anchored, insert_len = False, None, None, None, None
    if r1_hit is not None and end_class in ("proper", "r1_only", "r1_r1"):
        bead = bead_check(oseq, r1_start)
        bead_ok, bead_shift = bead.ok, bead.shift
        if bead.ok:
            polyT_len, polyT_anchored = polyt_length(oseq, bead.polyT_anchor), True
            cdna_start = bead.polyT_anchor
        else:
            polyT_len, polyT_anchored = polyt_fallback(oseq, r1_end), False
            cdna_start = r1_end
        if end_class == "proper":
            L = len(seq)
            if tso_side == "tail":           # oseq == seq, TSO' at the tail
                tso_inner = _first(tail_hits, "TSO", "-").start
            else:                            # oseq == revcomp(seq), TSO was "+" in the head
                tso_inner = L - _first(head_hits, "TSO", "+").end
            insert_len = tso_inner - cdna_start

    return SegmentCall(
        end_class=end_class,
        tso_side=tso_side,
        head_hits=tuple(head_hits),
        tail_hits=tuple(tail_hits),
        internal_hits=tuple(interior_hits),
        internal=any(h.kind in ("TSO", "R1") for h in interior_hits),
        bead_ok=bead_ok,
        bead_shift=bead_shift,
        polyT_len=polyT_len,
        polyT_anchored=polyT_anchored,
        insert_len=insert_len,
    )


def tso_toward_lower(call: SegmentCall, dl: int, dr: int) -> Optional[bool]:
    """True if the TSO end of a proper segment faces the lower-index adapter min(dl, dr).

    The head window faces dl in read coordinates.
    """
    if call.end_class != "proper":
        return None
    return (call.tso_side == "head") == (dl < dr)
