"""Reading BAM files record by record, with the ZMW filter the study and the reader command use."""
from collections.abc import Callable, Iterator
from pathlib import Path

import pysam


def iter_bam(path: str | Path, threads: int = 4, *, keep: Callable[[int], bool] | None = None,
             max_records: int | None = None) -> Iterator[pysam.AlignedSegment]:
    """Yield the records of an unaligned PacBio BAM in file order.

    keep: yield only records whose ZMW (tag zm) passes keep(zm). The file is read to the end, since a hash sample is
    spread over the whole file; S-reads of one ZMW are consecutive, so each ZMW is hashed once.
    max_records: stop after that many records (counted before the keep filter, as in the study's original reader).
    """
    with pysam.AlignmentFile(str(path), "rb", check_sq=False, threads=threads) as bam:
        last, last_ok = None, False
        for i, record in enumerate(bam):
            if max_records is not None and i >= max_records:
                return
            if keep is not None:
                zm = record.get_tag("zm")
                if zm != last:
                    last, last_ok = zm, keep(zm)
                if not last_ok:
                    continue
            yield record
