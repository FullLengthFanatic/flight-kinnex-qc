"""Full-read annotator, adapter chains and chain patterns."""
from typing import Optional

import edlib

from .patterns import FAIL_MAX_EDITS, MAS_ORDER, Pattern, all_patterns, revcomp
from .scan import Hit, find_all, windows

MAS_FLANK = 4           # a MAS hit is kept only if a TSO or R1 hit lies within this many bp (true adapters abut: 0 to 2 bp)
MAX_OVERLAP = 3         # hits overlapping by at most this many bp do not conflict (edlib may extend an end by a base)
MIN_EDGE_STRETCH = 50   # leading/trailing stretches shorter than this are not segments
PRIORITY = ("mirror", "skipped_slot", "repeated", "truncated", "complete", "unresolved")
HIFI_PATTERNS = all_patterns()
FAIL_PATTERNS = all_patterns(FAIL_MAX_EDITS)     # fail reads (1 to 2 passes)


def annotate_read(seq: str, patterns: list[Pattern] = HIFI_PATTERNS) -> list[Hit]:
    """All MAS, TSO and R1 hits along a whole read, sorted by start.

    Every pattern and strand is searched over the full read with edlib (find_all, exact up to
    each pattern's max_edits; HIFI_PATTERNS by default, FAIL_PATTERNS for fail reads). Hits
    overlapping by more than MAX_OVERLAP bp are resolved by lowest edits. In an array each adapter
    joins two molecule ends, so a MAS hit is kept only if a TSO or R1 hit lies within MAS_FLANK bp
    of it; this drops adapter-like matches inside cDNA.
    """
    L = len(seq)
    found = []
    for p in patterns:
        for strand in ("+", "-"):
            query = p.seq if strand == "+" else revcomp(p.seq)
            if edlib.align(query, seq, mode="HW", task="distance", k=p.max_edits)["editDistance"] < 0:
                continue
            found += find_all(seq, p, strand, 0, L)
    kept: list[Hit] = []
    for h in sorted(found, key=lambda x: (x.edits, x.start)):
        if all(min(h.end, k.end) - max(h.start, k.start) <= MAX_OVERLAP for k in kept):
            kept.append(h)
    primers = [h for h in kept if h.kind != "MAS"]
    out = [h for h in kept if h.kind != "MAS" or any(
        abs(h.start - q.end) <= MAS_FLANK or abs(q.start - h.end) <= MAS_FLANK for q in primers)]
    return sorted(out, key=lambda x: x.start)


def _token(h: Hit) -> str:
    return h.name if h.strand == "+" else h.name + "'"


def adapter_chain(hits: list[Hit]) -> tuple[list[str], bool]:
    """MAS tokens in read order ("B", "C'"), normalised so the majority strand is "+".

    If more tokens are "-" than "+", the order is reversed and every strand flipped.
    """
    tokens = [_token(h) for h in hits if h.kind == "MAS"]
    minus = sum(t.endswith("'") for t in tokens)
    if minus > len(tokens) - minus:
        return [t[:-1] if t.endswith("'") else t + "'" for t in reversed(tokens)], True
    return tokens, False


def chain_flags(tokens: list[str]) -> set[str]:
    """Flags of a normalised chain."""
    if not tokens:
        return {"unresolved"}
    letter = [t.rstrip("'") for t in tokens]
    minus = [t.endswith("'") for t in tokens]
    idx = [MAS_ORDER.index(x) for x in letter]
    pairs = list(zip(range(len(tokens) - 1), range(1, len(tokens))))
    flags = set()
    if tokens == MAS_ORDER:
        flags.add("complete")
    ordered = not any(minus) and all(idx[j] == idx[i] + 1 for i, j in pairs)
    if ordered and tokens != MAS_ORDER:
        flags.add("truncated")
    # a "+" run reads A to Q, a "-" run reads Q' down to A'
    if any(minus[i] == minus[j] and (idx[i] - idx[j] if minus[i] else idx[j] - idx[i]) > 1 for i, j in pairs):
        flags.add("skipped_slot")
    if any(letter[i] == letter[j] and minus[i] != minus[j] for i, j in pairs):
        flags.add("mirror")
    if len(set(tokens)) < len(tokens):
        flags.add("repeated")
    return flags or {"unresolved"}


def primary_pattern(flags: set[str]) -> str:
    """First of mirror, skipped_slot, repeated, truncated, complete, unresolved present in flags."""
    return next(p for p in PRIORITY if p in flags)


def split_segments(seq_len: int, hits: list[Hit]) -> list[tuple[int, int, Optional[str], Optional[str]]]:
    """(start, end, left_token, right_token) for each stretch between consecutive MAS hits.

    Leading and trailing stretches are included (token None) if at least MIN_EDGE_STRETCH bp.
    Tokens are in raw read orientation ("C" for "+", "C'" for "-").
    """
    mas = [h for h in hits if h.kind == "MAS"]
    if not mas:
        return [(0, seq_len, None, None)] if seq_len >= MIN_EDGE_STRETCH else []
    out = []
    if mas[0].start >= MIN_EDGE_STRETCH:
        out.append((0, mas[0].start, None, _token(mas[0])))
    for a, b in zip(mas, mas[1:]):
        if b.start > a.end:
            out.append((a.end, b.start, _token(a), _token(b)))
    if seq_len - mas[-1].end >= MIN_EDGE_STRETCH:
        out.append((mas[-1].end, seq_len, _token(mas[-1]), None))
    return out


def segment_hits(hits: list[Hit], start: int, end: int) -> tuple[list[Hit], list[Hit], list[Hit]]:
    """Read-level hits of the segment [start, end), shifted to segment coordinates and split into the
    head, tail and interior windows of scan.windows(end - start) by their midpoint.

    A MAS hit belongs to the segment only if it lies fully inside. A TSO or R1 hit belongs to it when
    its midpoint does: annotate_read lets a primer overlap the adjacent adapter by up to MAX_OVERLAP bp
    and segments are cut at the adapter, so such a primer runs past the segment edge. It is shifted
    without clipping (start below 0 or end above end - start); call_from_hits uses only a primer's
    inner end and the Read1 start, and both stay correct.
    """
    L = end - start
    head, tail, interior = windows(L)
    out = ([], [], [])
    for h in hits:
        if h.kind == "MAS":
            if h.start < start or h.end > end:
                continue
        elif not start <= (h.start + h.end) / 2 < end:
            continue
        x = Hit(h.name, h.kind, h.strand, h.start - start, h.end - start, h.edits)
        mid = (x.start + x.end) / 2
        if mid < head[1]:
            out[0].append(x)
        elif mid >= tail[0]:
            out[1].append(x)
        else:
            out[2].append(x)
    return out
