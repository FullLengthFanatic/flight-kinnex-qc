"""Edlib window scans, anchored polyT, bead check."""
from dataclasses import dataclass
from typing import Optional

import edlib

from .patterns import (
    CLS_LEN, LINKER1, LINKER1_OFFSET, LINKER1_MAX_SHIFT, LINKER2,
    LINKER2_AFTER_LINKER1, LINKER_MAX_MISMATCH, MAS_ORDER, POLYT_FALLBACK_SEARCH,
    POLYT_MAX_NONT, POLYT_START_SLACK, POLYT_WINDOW, END_WINDOW,
    SHORT_SEGMENT, UMI_LEN, Pattern, all_patterns, revcomp
)


@dataclass(frozen=True)
class Hit:
    """Pattern match result."""
    name: str  # Pattern name
    kind: str  # "MAS" | "TSO" | "R1"
    strand: str  # "+" or "-"
    start: int  # 0-based, half-open, full sequence coordinates
    end: int
    edits: int


def windows(L: int) -> tuple[tuple[int, int], tuple[int, int], Optional[tuple[int, int]]]:
    """
    Return head, tail, interior window spans for a segment of length L.
    Segments >= 300 bp use fixed 150 bp windows; shorter segments split at midpoint.
    """
    if L >= SHORT_SEGMENT:
        head = (0, END_WINDOW)
        tail = (L - END_WINDOW, L)
        interior = (END_WINDOW, L - END_WINDOW)
    else:
        mid = L // 2
        head = (0, mid)
        tail = (mid, L)
        interior = None
    return head, tail, interior


def best_hit(seq: str, p: Pattern, strand: str, lo: int, hi: int) -> Optional[Hit]:
    """
    Find best (minimum edits) hit of pattern p in seq[lo:hi].
    strand: "+" searches p; "-" searches revcomp(p).
    Returns Hit or None if edits > p.max_edits.
    Uses edlib HW (Hybrid) mode for local alignment.
    """
    search_seq = seq[lo:hi]
    query_seq = p.seq if strand == "+" else revcomp(p.seq)

    result = edlib.align(query_seq, search_seq, mode="HW", task="locations", k=p.max_edits)

    if result["editDistance"] < 0 or result["editDistance"] > p.max_edits:
        return None

    # HW returns locations of the best alignment(s); take the first
    if not result["locations"]:
        return None

    start_in_ref, end_in_ref = result["locations"][0]
    # Convert back to full sequence coordinates
    start = lo + start_in_ref
    end = lo + end_in_ref + 1  # edlib end is inclusive; we want half-open

    return Hit(
        name=p.name,
        kind=p.kind,
        strand=strand,
        start=start,
        end=end,
        edits=result["editDistance"]
    )


def find_all(seq: str, p: Pattern, strand: str, lo: int, hi: int) -> list[Hit]:
    """
    Repeatedly find best hits, masking found hits with N.
    Interior is scanned with masked hits to find all hits under threshold.
    """
    hits = []
    search_seq = list(seq[lo:hi])
    query_seq = p.seq if strand == "+" else revcomp(p.seq)

    while True:
        search_str = "".join(search_seq)
        result = edlib.align(query_seq, search_str, mode="HW", task="locations", k=p.max_edits)

        if result["editDistance"] < 0 or result["editDistance"] > p.max_edits or not result["locations"]:
            break

        start_in_ref, end_in_ref = result["locations"][0]
        start = lo + start_in_ref
        end = lo + end_in_ref + 1

        hits.append(Hit(
            name=p.name,
            kind=p.kind,
            strand=strand,
            start=start,
            end=end,
            edits=result["editDistance"]
        ))

        # Mask the hit with N
        for i in range(start_in_ref, end_in_ref + 1):
            search_seq[i] = "N"

    return hits


def scan_segment(seq: str, patterns: Optional[list[Pattern]] = None) -> tuple[list[Hit], list[Hit], list[Hit]]:
    """Scan one segment.

    For each pattern and strand, every hit under threshold is found once over the whole
    segment (find_all), then assigned to the window holding its midpoint: head, tail or
    interior, the same rule as annotate.segment_hits. Head and tail keep only the best hit
    per pattern and strand, fewest edits and ties broken by lower start; the interior keeps
    every hit.
    Returns (head_hits, tail_hits, interior_hits), each sorted by start.
    """
    if patterns is None:
        patterns = all_patterns()
    L = len(seq)
    head_span, tail_span, _ = windows(L)
    head_hits, tail_hits, interior_hits = [], [], []
    for p in patterns:
        for strand in ("+", "-"):
            # task="distance" is cheaper than find_all's first search (task="locations") over the
            # same sequence; when it finds nothing under threshold, find_all would return no hits.
            query = p.seq if strand == "+" else revcomp(p.seq)
            if edlib.align(query, seq, mode="HW", task="distance", k=p.max_edits)["editDistance"] < 0:
                continue
            head_best: Optional[Hit] = None
            tail_best: Optional[Hit] = None
            for x in find_all(seq, p, strand, 0, L):
                mid = (x.start + x.end) / 2
                if mid < head_span[1]:
                    if head_best is None or (x.edits, x.start) < (head_best.edits, head_best.start):
                        head_best = x
                elif mid >= tail_span[0]:
                    if tail_best is None or (x.edits, x.start) < (tail_best.edits, tail_best.start):
                        tail_best = x
                else:
                    interior_hits.append(x)
            if head_best is not None:
                head_hits.append(head_best)
            if tail_best is not None:
                tail_hits.append(tail_best)
    for hits in (head_hits, tail_hits, interior_hits):
        hits.sort(key=lambda x: x.start)
    return head_hits, tail_hits, interior_hits


def encode_hits(hits: list[Hit]) -> str:
    """Encode hits as string: 'name:strand:start:end:edits;...'."""
    if not hits:
        return ""
    parts = []
    for h in hits:
        parts.append(f"{h.name}:{h.strand}:{h.start}:{h.end}:{h.edits}")
    return ";".join(parts)


def decode_hits(s: str) -> list[Hit]:
    """Decode hits from string (inverse of encode_hits)."""
    if not s:
        return []

    hits = []
    for part in s.split(";"):
        name, strand, start, end, edits = part.split(":")
        # Recover kind from name
        if name in MAS_ORDER:
            kind = "MAS"
        elif name == "TSO":
            kind = "TSO"
        elif name == "R1":
            kind = "R1"
        else:
            kind = "unknown"

        hits.append(Hit(
            name=name,
            kind=kind,
            strand=strand,
            start=int(start),
            end=int(end),
            edits=int(edits)
        ))

    return hits


@dataclass(frozen=True)
class BeadCall:
    """Result of bead_check."""
    ok: bool
    shift: Optional[int]  # 0..3 if ok, else None
    polyT_anchor: Optional[int]  # Index in oseq if ok, else None


def bead_check(oseq: str, r1_start: int) -> BeadCall:
    """
    Check for bead-side linker pattern in oseq (oriented so R1 reads "+" near start).
    GTGA at 29..32 nt after R1 start (shift 0..3), GACA exactly 13 nt after GTGA start,
    CLS3 (9 nt) and UMI (8 nt) follow at fixed lengths.
    Returns BeadCall with ok status, shift, and polyT anchor.
    """
    for shift in range(LINKER1_MAX_SHIFT + 1):
        linker1_pos = r1_start + LINKER1_OFFSET + shift
        if linker1_pos + len(LINKER1) > len(oseq):
            continue

        # Check LINKER1 with up to 1 mismatch
        linker1_candidate = oseq[linker1_pos:linker1_pos + len(LINKER1)]
        mismatch_count = sum(1 for a, b in zip(LINKER1, linker1_candidate) if a != b)

        if mismatch_count > LINKER_MAX_MISMATCH:
            continue

        # Check LINKER2 at fixed distance
        linker2_pos = linker1_pos + LINKER2_AFTER_LINKER1
        if linker2_pos + len(LINKER2) > len(oseq):
            continue

        linker2_candidate = oseq[linker2_pos:linker2_pos + len(LINKER2)]
        mismatch_count = sum(1 for a, b in zip(LINKER2, linker2_candidate) if a != b)

        if mismatch_count > LINKER_MAX_MISMATCH:
            continue

        # Both linkers match; polyT anchor is after CLS3 and UMI
        polyT_anchor = linker2_pos + len(LINKER2) + CLS_LEN + UMI_LEN
        return BeadCall(ok=True, shift=shift, polyT_anchor=polyT_anchor)

    return BeadCall(ok=False, shift=None, polyT_anchor=None)


def polyt_length(oseq: str, start: int) -> int:
    """
    Measure continuous polyT length starting from anchor position.
    Start at first T within POLYT_START_SLACK nt of start.
    Extend while every 8 nt window has at most 1 non-T.
    Trim trailing non-T. No minimum.
    """
    if start >= len(oseq):
        return 0

    # Find first T within POLYT_START_SLACK
    t_pos = -1
    for i in range(start, min(start + POLYT_START_SLACK + 1, len(oseq))):
        if oseq[i] == "T":
            t_pos = i
            break

    if t_pos == -1:
        return 0

    # Extend while T-dominant (every 8 nt window has <= 1 non-T)
    end = t_pos
    while end < len(oseq):
        window_start = max(t_pos, end - POLYT_WINDOW + 1)
        window_end = min(end + 1, len(oseq))
        window = oseq[window_start:window_end]
        non_t_count = sum(1 for c in window if c != "T")

        if non_t_count <= POLYT_MAX_NONT:
            end += 1
        else:
            break

    # Trim trailing non-T
    while end > t_pos and oseq[end - 1] != "T":
        end -= 1

    return end - t_pos


def polyt_fallback(oseq: str, r1_end: int) -> int:
    """
    Fallback polyT measurement: longest T-dominant stretch starting within
    POLYT_FALLBACK_SEARCH nt after r1_end.
    """
    if r1_end >= len(oseq):
        return 0

    search_start = r1_end
    search_end = min(r1_end + POLYT_FALLBACK_SEARCH, len(oseq))

    max_length = 0

    # Try starting position for each T in the search range
    for start in range(search_start, search_end):
        if oseq[start] != "T":
            continue

        length = polyt_length(oseq, start)
        max_length = max(max_length, length)

    return max_length
