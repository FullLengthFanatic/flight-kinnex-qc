import multiprocessing
import os
import pytest
import signal
import subprocess
import sys
import textwrap
import time
import pyarrow as pa
import pyarrow.parquet as pq

from kinnex_qc.parallel import batched, bounded_imap, process_pool, ParquetBatchWriter
from kinnex_qc.schemas import SEGMENTS


class TestBatched:
    """Test batching function."""

    def test_batched_basic(self):
        items = list(range(25))
        batches = list(batched(items, 10))
        assert len(batches) == 3
        assert batches[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert batches[1] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert batches[2] == [20, 21, 22, 23, 24]

    def test_batched_empty(self):
        batches = list(batched([], 10))
        assert batches == []

    def test_batched_exact(self):
        items = list(range(20))
        batches = list(batched(items, 10))
        assert len(batches) == 2
        assert batches[1] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


class TestParquetBatchWriter:
    """Test ParquetBatchWriter round trip."""

    def test_round_trip(self, tmp_path):
        parquet_path = tmp_path / "test.parquet"

        schema = SEGMENTS
        writer = ParquetBatchWriter(parquet_path, schema)

        # Write first batch
        batch1 = {
            "zm": [100, 101, 102],
            "di": [0, 1, 2],
            "dl": [0, 1, 2],
            "dr": [1, 2, 3],
            "pos": [0, 1, 2],
            "direction": [1, 1, 1],
            "qs": [0, 100, 200],
            "qe": [100, 200, 300],
            "sread_len": [100, 100, 100],
            "np": [3, 3, 3],
            "rq": [0.99, 0.99, 0.99],
            "end_class": ["proper", "proper", "proper"],
            "tso_side": ["head", "head", "head"],
            "tso_toward_lower": [True, True, True],
            "internal": [False, False, False],
            "n_internal": [0, 0, 0],
            "internal_hits": ["", "", ""],
            "head_hits": ["", "", ""],
            "tail_hits": ["", "", ""],
            "bead_ok": [True, True, True],
            "bead_shift": [0, 0, 0],
            "polyT_len": [10, 10, 10],
            "polyT_anchored": [True, True, True],
            "insert_len": [100, 100, 100],
        }
        writer.write(batch1)

        # Write second batch
        batch2 = {
            "zm": [103, 104],
            "di": [3, 4],
            "dl": [3, 4],
            "dr": [4, 5],
            "pos": [3, 4],
            "direction": [1, 1],
            "qs": [300, 400],
            "qe": [400, 500],
            "sread_len": [100, 100],
            "np": [3, 3],
            "rq": [0.99, 0.99],
            "end_class": ["proper", "proper"],
            "tso_side": ["head", "head"],
            "tso_toward_lower": [True, True],
            "internal": [False, False],
            "n_internal": [0, 0],
            "internal_hits": ["", ""],
            "head_hits": ["", ""],
            "tail_hits": ["", ""],
            "bead_ok": [True, True],
            "bead_shift": [0, 0],
            "polyT_len": [10, 10],
            "polyT_anchored": [True, True],
            "insert_len": [100, 100],
        }
        writer.write(batch2)
        writer.close()

        # Read back and verify
        table = pq.read_table(parquet_path)
        assert len(table) == 5
        assert table.column("zm").to_pylist() == [100, 101, 102, 103, 104]

    def test_row_groups_of_100k_rows(self, tmp_path):
        schema = pa.schema([("zm", pa.int64()), ("x", pa.float64())])
        batches = [{"zm": list(range(i * 10_000, (i + 1) * 10_000)), "x": [i / 3] * 10_000} for i in range(25)]
        with ParquetBatchWriter(tmp_path / "new.parquet", schema) as writer:
            for b in batches:
                writer.write(b)
        # the previous writer: one row group per batch
        with pq.ParquetWriter(str(tmp_path / "before.parquet"), schema) as before:
            for b in batches:
                before.write_batch(pa.RecordBatch.from_pydict(b, schema=schema))
        assert pq.ParquetFile(tmp_path / "new.parquet").num_row_groups == 3
        assert pq.ParquetFile(tmp_path / "before.parquet").num_row_groups == 25
        new = pq.read_table(tmp_path / "new.parquet")
        assert new.equals(pq.read_table(tmp_path / "before.parquet"))
        assert new["zm"].to_pylist() == list(range(250_000))

    def test_close_without_write_gives_an_empty_table(self, tmp_path):
        """An empty BAM: write() is never called, but a reader should get an empty table, not a missing file."""
        schema = pa.schema([("zm", pa.int64()), ("x", pa.float64())])
        with ParquetBatchWriter(tmp_path / "empty.parquet", schema):
            pass
        table = pq.read_table(tmp_path / "empty.parquet")
        assert table.schema == schema and table.num_rows == 0


def run_script(tmp_path, code: str, *args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    """Run a script in a fresh interpreter; a hang fails the test through TimeoutExpired."""
    script = tmp_path / "script.py"
    script.write_text(textwrap.dedent(code))
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=timeout)


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")   # numpy's BLAS threads
class TestBoundedImap:
    """bounded_imap on process_pool: order, bounded input, and failure on a dead worker."""

    def test_in_order_and_bounded(self):
        consumed = []

        def items():
            for i in range(40):
                consumed.append(i)
                yield -i

        with process_pool(2) as pool:
            it = bounded_imap(pool, abs, items(), 4)
            first = next(it)
            assert len(consumed) == 4
            assert [first] + list(it) == list(range(40))

    def test_process_pool_starts_every_worker(self):
        before = set(multiprocessing.active_children())
        with process_pool(3):
            assert len(set(multiprocessing.active_children()) - before) == 3

    def test_killed_worker_raises(self, tmp_path):
        r = run_script(tmp_path, """
            import os, signal, sys
            from concurrent.futures.process import BrokenProcessPool
            from kinnex_qc.parallel import bounded_imap, process_pool

            def work(x):
                if x == 5:
                    os.kill(os.getpid(), signal.SIGKILL)
                return x

            with process_pool(2) as pool:
                try:
                    list(bounded_imap(pool, work, range(50), 8))
                except BrokenProcessPool:
                    sys.exit(3)
            """)
        assert r.returncode == 3, r.stderr

    def test_workers_die_with_killed_parent(self, tmp_path):
        """A killed stage parent must not leave its pool workers running (PR_SET_PDEATHSIG), holding
        the memory they forked. A timeout here means the fix regressed, not a hang."""
        pids_path = tmp_path / "pids.txt"
        script = tmp_path / "parent.py"
        script.write_text(textwrap.dedent(f"""
            import os, time
            from kinnex_qc.parallel import process_pool
            with process_pool(3) as pool:
                pids = [f.result() for f in [pool.submit(os.getpid) for _ in range(3)]]
                with open({str(pids_path)!r}, "w") as f:
                    f.write(" ".join(map(str, pids)))
                time.sleep(300)
            """))
        p = subprocess.Popen([sys.executable, str(script)])
        try:
            deadline = time.time() + 30
            while not pids_path.exists():
                assert time.time() < deadline and p.poll() is None
                time.sleep(0.1)
            pids = [int(x) for x in pids_path.read_text().split()]
            os.kill(p.pid, signal.SIGKILL)
            p.wait(timeout=5)
            deadline = time.time() + 5
            while time.time() < deadline and any(os.path.exists(f"/proc/{pid}") for pid in pids):
                time.sleep(0.1)
            assert not any(os.path.exists(f"/proc/{pid}") for pid in pids), \
                f"workers {pids} still alive 5s after the parent was killed"
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
