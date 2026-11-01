"""`--write-molecules`: a BAM of the molecules a full-read segmentation finds, one unaligned record
per usable molecule, ready for lima --isoseq --peek-guess as if skera had written it. Port of the
study's downstream test's molecule-writing loop.
"""
from collections.abc import Callable
from pathlib import Path

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
    usable = molecules.filter(USABLE)
    by_zm: dict[int, list[tuple[int, int]]] = {}
    for zm, start, end in usable.select("zm", "start", "end").iter_rows():
        by_zm.setdefault(zm, []).append((start, end))

    with pysam.AlignmentFile(str(template), "rb", check_sq=False) as t:
        header = t.header.to_dict()
    rg = header["RG"][0]["ID"]

    def _keep(zm):
        return zm in by_zm and (keep is None or keep(zm))

    n = 0
    with pysam.AlignmentFile(str(out), "wb", header=header) as w:
        for r in iter_bam(hifi, keep=_keep):
            zm = r.get_tag("zm")
            seq, qual = r.query_sequence, r.query_qualities
            movie = r.query_name.split("/")[0]
            for di, (s, e) in enumerate(sorted(by_zm[zm])):
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
