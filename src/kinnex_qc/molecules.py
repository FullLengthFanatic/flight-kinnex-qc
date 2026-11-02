"""`--write-molecules`: a BAM of the molecules a full-read segmentation finds, one unaligned record
per usable molecule, ready for lima --isoseq --peek-guess as if skera had written it. Port of the
study's downstream test's molecule-writing loop.
"""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pysam

from .io import iter_bam

USABLE = (pl.col("end_class") == "proper") & ~pl.col("internal")
"""Molecules usable by --write-molecules and the slot budget (skera.budget_classes): proper end_class, not internal (fused)."""


def write_molecules(hifi: Path, molecules: pl.DataFrame, template: Path, out: Path,
                     keep: Callable[[int], bool] | None = None) -> int:
    """Write one unaligned record per usable molecule of each read in `molecules` to `out`.

    Record name movie/zm/ccs/start_end (movie: the HiFi record's name before its first '/'), sequence
    and qualities sliced [start:end], tags zm, np, rq copied from the HiFi record when present, di the
    molecule's index among its read's written molecules sorted by start, qs and qe the slice bounds,
    RG the template's first read group. header is the template's header (to_dict()), unchanged.

    write_molecules keeps only the usable molecules of `molecules` itself (callers pass every
    molecule). keep(zm), when given, further restricts which reads are written. The HiFi BAM is
    read once, in file order, so the output is in HiFi file order. Returns the number of records written.
    """
    # A read's molecules are found by binary search in columns sorted by (zm, start, end), not in a dict of Python
    # (start, end) tuples, which take about 140 bytes per molecule.
    usable = molecules.filter(USABLE).select("zm", "start", "end").sort("zm", "start", "end")
    zms, starts, ends = (usable[c].to_numpy() for c in ("zm", "start", "end"))

    def _rows(zm: int) -> tuple[int, int]:
        """zm's rows [lo, hi) in the sorted columns (lo == hi: no usable molecule). zm is cast to the column's dtype,
        since a search with another dtype converts the whole column on every call."""
        key = zms.dtype.type(zm)
        return int(np.searchsorted(zms, key, "left")), int(np.searchsorted(zms, key, "right"))

    with pysam.AlignmentFile(str(template), "rb", check_sq=False) as t:
        header = t.header.to_dict()
    rg = header["RG"][0]["ID"]

    def _keep(zm):
        lo, hi = _rows(zm)
        return lo < hi and (keep is None or keep(zm))

    n = 0
    with pysam.AlignmentFile(str(out), "wb", header=header) as w:
        for r in iter_bam(hifi, keep=_keep):
            zm = r.get_tag("zm")
            seq, qual = r.query_sequence, r.query_qualities
            movie = r.query_name.split("/")[0]
            lo, hi = _rows(zm)
            for di, (s, e) in enumerate(zip(starts[lo:hi].tolist(), ends[lo:hi].tolist())):
                a = pysam.AlignedSegment(w.header)
                a.query_name = f"{movie}/{zm}/ccs/{s}_{e}"
                a.query_sequence, a.query_qualities = seq[s:e], qual[s:e]
                a.flag = 4
                for tag in ("zm", "np", "rq"):
                    if r.has_tag(tag):
                        a.set_tag(tag, r.get_tag(tag))
                a.set_tag("di", di)
                a.set_tag("qs", s)
                a.set_tag("qe", e)
                a.set_tag("RG", rg)
                w.write(a)
                n += 1
    return n
