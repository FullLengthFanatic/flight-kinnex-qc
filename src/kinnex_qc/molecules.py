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


def _usable_columns(molecules: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """zm, start and end of the usable molecules, sorted by (zm, start, end). A read's molecules are found in them by
    binary search (_rows), not in a dict of Python (start, end) tuples, which take about 140 bytes per molecule."""
    usable = molecules.filter(USABLE).select("zm", "start", "end").sort("zm", "start", "end")
    return tuple(usable[c].to_numpy() for c in ("zm", "start", "end"))


def _rows(zms: np.ndarray, zm: int) -> tuple[int, int]:
    """zm's rows [lo, hi) in the sorted zms (lo == hi: no usable molecule). zm is cast to the column's dtype, since a
    search with another dtype converts the whole column on every call."""
    key = zms.dtype.type(zm)
    return int(np.searchsorted(zms, key, "left")), int(np.searchsorted(zms, key, "right"))


class MoleculeSink:
    """molecules.bam, written read by read: write_read builds and writes the records of one HiFi read, add those of a
    batch of reads. runner.annotate hands each batch of HiFi records to add as soon as the workers return it, so a
    molecules.bam of every read holds no more in memory than the batches in flight.

    Record name movie/zm/ccs/start_end (movie: the HiFi record's name before its first '/'), flag 4, sequence and
    qualities sliced [start:end], tags zm, np, rq copied from the HiFi record when present, di the molecule's index
    among its read's written molecules sorted by start, qs and qe the slice bounds, RG the template's first read
    group. The header is the template's header (to_dict()), unchanged. The output is opened at the first write (or at
    close, for a BAM of no records), not here, so a sink made before annotate() forks its workers is not open in them.

    The records go to out with ".part" appended, renamed to out by close, when the with-block ends without an error. A
    with-block that raises (a killed worker, Ctrl-C) closes the writer and deletes the .part file, so a failed run
    leaves no truncated BAM that looks complete.
    """

    def __init__(self, template: Path, out: Path):
        with pysam.AlignmentFile(str(template), "rb", check_sq=False) as t:
            self._header = t.header.to_dict()
        self._rg = self._header["RG"][0]["ID"]
        self._out = Path(out)
        self._part = self._out.with_name(self._out.name + ".part")
        self._w = None
        self.written = 0

    def _writer(self) -> pysam.AlignmentFile:
        if self._w is None:
            self._w = pysam.AlignmentFile(str(self._part), "wb", header=self._header)
        return self._w

    def write_read(self, r: pysam.AlignedSegment, starts: np.ndarray, ends: np.ndarray) -> None:
        """Write HiFi record r's molecules [starts[i], ends[i]), given sorted by start (di follows that order)."""
        w = self._writer()
        zm = r.get_tag("zm")
        seq, qual = r.query_sequence, r.query_qualities
        movie = r.query_name.split("/")[0]
        for di, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
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
            a.set_tag("RG", self._rg)
            w.write(a)
            self.written += 1

    def add(self, records: list[pysam.AlignedSegment], molecules: pl.DataFrame) -> None:
        """Write the usable molecules of each HiFi record in records, in the records' order; molecules: the molecule
        rows (ANNOT_SEGMENTS columns) of these records, every end class (the usable ones are picked here)."""
        zms, starts, ends = _usable_columns(molecules)
        for r in records:
            lo, hi = _rows(zms, r.get_tag("zm"))
            if lo < hi:
                self.write_read(r, starts[lo:hi], ends[lo:hi])

    def close(self) -> None:
        self._writer().close()
        self._part.replace(self._out)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()
            return
        if self._w is not None:
            self._w.close()
        self._part.unlink(missing_ok=True)


def write_molecules(hifi: Path, molecules: pl.DataFrame, template: Path, out: Path,
                     keep: Callable[[int], bool] | None = None) -> int:
    """Write one unaligned record per usable molecule of each read in `molecules` to `out` (record fields: see
    MoleculeSink).

    write_molecules keeps only the usable molecules of `molecules` itself (callers pass every
    molecule). keep(zm), when given, further restricts which reads are written. The HiFi BAM is
    read once, in file order, so the output is in HiFi file order. Returns the number of records written.
    """
    zms, starts, ends = _usable_columns(molecules)

    def _keep(zm):
        lo, hi = _rows(zms, zm)
        return lo < hi and (keep is None or keep(zm))

    with MoleculeSink(template, out) as sink:
        for r in iter_bam(hifi, keep=_keep):
            lo, hi = _rows(zms, r.get_tag("zm"))
            sink.write_read(r, starts[lo:hi], ends[lo:hi])
    return sink.written
