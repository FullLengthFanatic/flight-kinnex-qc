import collections
import ctypes
import os
import signal
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Iterator, Iterable, TypeVar
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

T = TypeVar('T')


def batched(it: Iterable[T], n: int) -> Iterator[list[T]]:
    """Batch an iterable into chunks of size n."""
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def _die_with_parent(parent_pid: int) -> None:
    """Worker initializer: SIGKILL this worker if its parent is gone or dies later.

    multiprocessing.Pool workers close their copy of the task pipe's write end, so a dead
    parent gives them EOF and they exit. ProcessPoolExecutor workers keep both ends of the
    call queue and block on it forever, so an OOM-killed or SIGKILLed parent leaves them
    running, holding their forked copy-on-write memory. PR_SET_PDEATHSIG makes the kernel
    SIGKILL this process when its parent dies; os.getppid() is checked right after in case
    the parent died between the fork and this call, before the signal was armed.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    if os.getppid() != parent_pid:
        os._exit(1)


def process_pool(processes: int) -> ProcessPoolExecutor:
    """A fork ProcessPoolExecutor whose workers have all started when it is returned.

    A killed worker (OOM killer) makes the pending results raise BrokenProcessPool, where
    multiprocessing.Pool waits forever. Fork, so the workers see the parent's module globals.
    The executor forks at its first submit; forking here instead keeps the tar and samtools
    pipes the caller opens later out of the workers. A worker holding such a pipe would keep
    samtools from getting SIGPIPE when the parent stops reading early (pilot), and hang it.
    Workers die with this process (see _die_with_parent), so a killed parent does not leave
    them mapping its forked memory.
    """
    parent_pid = os.getpid()
    pool = ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("fork"),
                               initializer=_die_with_parent, initargs=(parent_pid,))
    for f in [pool.submit(os.getpid) for _ in range(processes)]:
        f.result()
    return pool


def bounded_imap(pool: ProcessPoolExecutor, func, items: Iterable, max_inflight: int) -> Iterator:
    """Like pool.map, in order, but with at most max_inflight tasks pending.

    pool.map submits its whole input at once, which would pull a whole BAM into memory
    when reading is faster than the workers. A dead worker raises BrokenProcessPool here.
    On the parent side, when the caller stops early or fails: tasks not yet handed to a
    worker are cancelled, but up to processes + 1 tasks already queued still run, so a
    failure can take a few task lengths to surface.
    """
    pending = collections.deque()
    try:
        for item in items:
            pending.append(pool.submit(func, item))
            if len(pending) >= max_inflight:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for f in pending:
            f.cancel()


class ParquetBatchWriter:
    """Write a parquet file in batches, one row group per row_group_rows or more buffered rows."""

    def __init__(self, path: Path, schema: pa.Schema, row_group_rows: int = 100_000):
        self.path = Path(path)
        self.schema = schema
        self.row_group_rows = row_group_rows
        self.writer = None
        self.buffer = []
        self.buffered = 0

    def write(self, columns: dict[str, list]) -> None:
        """Write a batch of data."""
        arrays = [pa.array(columns[field.name], type=field.type) for field in self.schema]
        batch = pa.RecordBatch.from_arrays(arrays, schema=self.schema)

        if self.writer is None:
            self.writer = pq.ParquetWriter(str(self.path), self.schema)

        self.buffer.append(batch)
        self.buffered += batch.num_rows
        if self.buffered >= self.row_group_rows:
            self.flush()

    def flush(self) -> None:
        """Write the buffered batches as one row group."""
        if self.buffer:
            table = pa.Table.from_batches(self.buffer, schema=self.schema)
            self.writer.write_table(table, row_group_size=max(table.num_rows, 1))
            self.buffer, self.buffered = [], 0

    def close(self) -> None:
        """Write the buffered rows and close the writer; an empty file with the schema if write() was never called,
        so a reader of an empty BAM gets an empty table instead of a missing file."""
        if self.writer is None:
            self.writer = pq.ParquetWriter(str(self.path), self.schema)
        self.flush()
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
