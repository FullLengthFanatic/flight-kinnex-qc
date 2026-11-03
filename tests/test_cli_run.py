"""Tests for `kinnex-qc run`."""
import csv
import gzip
import json
import re
import subprocess
import tracemalloc
import warnings
from concurrent.futures.process import BrokenProcessPool

import polars as pl
import pyarrow as pa
import pysam
import pytest

from kinnex_qc import cli, runner
from kinnex_qc.annotate import FAIL_PATTERNS, HIFI_PATTERNS
from kinnex_qc.io import iter_bam
from kinnex_qc.parallel import batched
from kinnex_qc.patterns import KINNEX8_SC3
from kinnex_qc.sampling import in_hash_sample
from kinnex_qc.schemas import ANNOT_SEGMENTS, READ_FEATURES, READ_SUMMARY

# the pytest process carries threads from earlier tests; the CLI forks its pool workers before any
# BAM is open, which is fine, but multiprocessing warns about it when run in-process like this
_IGNORE_FORK_WARNING = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


def _run(*args, check=False):
    return subprocess.run(["kinnex-qc", "run", *args], capture_output=True, text=True, check=check)


def _reads_tsv(path):
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _sreads_bam_without_skera_pg(path):
    """A minimal S-read BAM whose header has a @PG line, but not one naming skera."""
    header = {"HD": {"VN": "1.6", "SO": "unknown"}, "PG": [{"ID": "other", "PN": "other", "VN": "1.0"}]}
    with pysam.AlignmentFile(str(path), "wb", header=header):
        pass
    return path


def test_default_chemistry_patterns_match_annotate_module():
    """For the default chemistry, runner's per-chemistry pattern cache must equal the study's module-level
    HIFI_PATTERNS and FAIL_PATTERNS (same thresholds, same adapters)."""
    assert runner._hifi_patterns(KINNEX8_SC3) == HIFI_PATTERNS
    assert runner._fail_patterns(KINNEX8_SC3) == FAIL_PATTERNS


@_IGNORE_FORK_WARNING
def test_annotate_molecule_end_classes(test_bams):
    """Sanity check of the test BAM fixture itself: the mirror junctions come out as built."""
    hifi, _ = test_bams
    read_summary, molecules, _ = runner.annotate(hifi, KINNEX8_SC3, 1.0, 2)
    assert read_summary.sort("zm")["primary_pattern"].to_list() == ["complete", "complete", "complete",
                                                                    "mirror", "mirror", "mirror"]
    tso_tso = molecules.filter(pl.col("zm").is_in([4, 5]) & (pl.col("left_token") == "H'") & (pl.col("right_token") == "H"))
    assert tso_tso["end_class"].to_list() == ["tso_tso", "tso_tso"]
    r1_r1 = molecules.filter((pl.col("zm") == 6) & (pl.col("left_token") == "B") & (pl.col("right_token") == "B'"))
    assert r1_r1["end_class"].to_list() == ["r1_r1"]
    # every proper molecule in the fixture is built so its TSO end faces the higher-index adapter,
    # as in real data (mirror arms included: both run from revcomp(molecule(...)))
    proper = molecules.filter(pl.col("end_class") == "proper")
    assert proper.height > 0
    assert not proper["tso_toward_lower"].any()


def _sample(fraction):
    return None if fraction >= 1 else (lambda zm: in_hash_sample(zm, fraction))


def _annotate_the_list_way(hifi, fraction):
    """annotate()'s three frames built the way it built them while it kept Python lists: annotate_batch over batches of
    runner.BATCH sampled reads (serially, in file order), every batch's column lists kept and merged into one list per
    column at the end, the merged lists made into one arrow table and sorted as annotate() sorts."""
    items = ((r.get_tag("zm"), "hifi", r.query_sequence, r.get_tag("np"), r.get_tag("rq"), True)
             for r in iter_bam(hifi, threads=1, keep=_sample(fraction)))
    parts = [runner.annotate_batch(b, KINNEX8_SC3) for b in batched(items, runner.BATCH)]

    def frame(schema, drop, k):
        fields = [f for f in schema if f.name not in drop]
        cols = {f.name: [v for part in parts for v in part[k][f.name]] for f in fields}
        return pl.from_arrow(pa.Table.from_arrays([pa.array(cols[f.name], type=f.type) for f in fields],
                                                  schema=pa.schema(fields)))

    return (frame(READ_SUMMARY, {"source", "skera_passing"}, 1).sort("zm"),
            frame(ANNOT_SEGMENTS, {"source"}, 2).sort(["zm", "seg_index"]),
            frame(READ_FEATURES, {"source"}, 0).sort(["zm", "start"]))


@_IGNORE_FORK_WARNING
@pytest.mark.parametrize(("batch", "fraction", "reads"), [(200, 1.0, 6), (1, 1.0, 6), (2, 0.5, None), (1, 0.000001, 0)])
def test_annotate_frames_equal_the_list_built_reference(test_bams, monkeypatch, batch, fraction, reads):
    """Keeping each batch as arrow must not change a value, a dtype or the row order: all reads in one batch, one
    read per batch, a hash sample over batches of 2, and a sample of zero reads."""
    hifi, _ = test_bams
    monkeypatch.setattr(runner, "BATCH", batch)
    got = runner.annotate(hifi, KINNEX8_SC3, fraction, 2)
    want = _annotate_the_list_way(hifi, fraction)
    assert reads is None or got[0].height == reads
    for g, w in zip(got, want):
        assert g.schema == w.schema
        assert g.equals(w)


class _FakeRecord:
    """Stands in for a pysam record in the memory tests: the tags and the sequence the readers ask for."""
    query_sequence = "ACGT"

    def __init__(self, tags):
        self.tags = tags

    def get_tag(self, name):
        return self.tags[name]


def _fake_annotate_batch(batch, chem=None):
    """annotate_batch's three column dicts without annotating anything: 2,000 feature rows per read (zm, start and
    end set, the rest null), one row each of read_summary and molecules (zm set, the rest null)."""
    n = 2000
    feats = {f.name: [] for f in READ_FEATURES}
    summ = {f.name: [] for f in READ_SUMMARY}
    segs = {f.name: [] for f in ANNOT_SEGMENTS}
    for item in batch:
        zm = item[0]
        for k in feats:
            feats[k] += {"zm": [zm] * n, "start": list(range(1000, 1000 + n)), "end": list(range(1016, 1016 + n))}.get(k, [None] * n)
        for cols in (summ, segs):
            for k in cols:
                cols[k].append(zm if k == "zm" else None)
    return feats, summ, segs


def _python_heap_peak(fn):
    """fn's result and the peak of the Python heap while it ran (tracemalloc: Python objects, numpy arrays; not
    arrow's or polars' own buffers)."""
    tracemalloc.start()
    try:
        result = fn()
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@_IGNORE_FORK_WARNING
def test_annotate_keeps_rows_as_arrow_not_python_lists(monkeypatch):
    """annotate() turns each batch's column lists into arrow as it arrives, so the Python heap holds a few batches at a
    time however many rows the BAM gives. 300 reads of 2,000 fake feature rows each, one read per batch: kept as
    Python lists to the end, the 600,000 rows took a Python heap peak of 122 MB (measured before the change; 2.4 MB after)."""
    records = [_FakeRecord({"zm": zm, "np": 10, "rq": 0.999}) for zm in range(1, 301)]
    monkeypatch.setattr(runner, "iter_bam", lambda path, threads=4, keep=None: iter(records))
    monkeypatch.setattr(runner, "annotate_batch", _fake_annotate_batch)
    monkeypatch.setattr(runner, "BATCH", 1)
    (read_summary, molecules, features), peak = _python_heap_peak(lambda: runner.annotate("unused.bam", KINNEX8_SC3, 1.0, 1))
    assert (read_summary.height, molecules.height, features.height) == (300, 300, 600_000)
    assert peak < 20 * 2**20


class _ListSink:
    """Keeps what annotate() hands a molecule sink: each batch's HiFi records and that batch's molecule rows."""

    def __init__(self):
        self.calls = []

    def add(self, records, molecules):
        self.calls.append((records, molecules))


@_IGNORE_FORK_WARNING
@pytest.mark.parametrize(("batch", "fraction"), [(200, 1.0), (2, 1.0), (2, 0.5), (1, 0.000001)])
def test_annotate_with_a_molecule_sink_keeps_the_sampled_rows_and_hands_it_every_read(test_bams, monkeypatch, batch,
                                                                                         fraction):
    """With a molecule sink, annotate() returns the same frames and hifi_reads_in_file as without one (the hash sample's
    rows), and hands the sink every record of the BAM, in file order, each batch with its own molecule rows."""
    hifi, _ = test_bams
    monkeypatch.setattr(runner, "BATCH", batch)
    want_counts, got_counts, sink = {}, {}, _ListSink()
    want = runner.annotate(hifi, KINNEX8_SC3, fraction, 2, counts=want_counts)
    got = runner.annotate(hifi, KINNEX8_SC3, fraction, 2, counts=got_counts, molecules_out=sink)
    for g, w in zip(got, want):
        assert g.schema == w.schema
        assert g.equals(w)
    assert got_counts == want_counts == {"hifi_reads_in_file": 6}

    file_order = [r.get_tag("zm") for r in pysam.AlignmentFile(str(hifi), "rb", check_sq=False)]
    assert [r.get_tag("zm") for records, _ in sink.calls for r in records] == file_order
    assert all(len(records) <= batch for records, _ in sink.calls)
    for records, mols in sink.calls:
        assert set(mols["zm"].to_list()) == {r.get_tag("zm") for r in records}
    every_read = runner.annotate(hifi, KINNEX8_SC3, 1.0, 2)[1]
    assert pl.concat([mols for _, mols in sink.calls]).sort(["zm", "seg_index"]).equals(every_read)


@_IGNORE_FORK_WARNING
def test_annotate_hands_the_sink_each_batch_as_it_comes_back(monkeypatch):
    """With a molecule sink, every record is read (no keep filter on the pass) and each batch of records goes to the
    sink as soon as its result comes back, so the parent holds the records of the batches in flight only (4 per
    worker), never the whole BAM: 300 reads, one per batch, one worker, a hash sample of half."""
    yielded = 0

    def fake_iter_bam(path, threads=4, keep=None):
        nonlocal yielded
        assert keep is None
        for zm in range(1, 301):
            yielded += 1
            yield _FakeRecord({"zm": zm, "np": 10, "rq": 0.999})

    received, held = [], []

    class Sink:
        def add(self, records, molecules):
            held.append(yielded - len(received))
            received.extend(r.get_tag("zm") for r in records)

    monkeypatch.setattr(runner, "iter_bam", fake_iter_bam)
    monkeypatch.setattr(runner, "annotate_batch", _fake_annotate_batch)
    monkeypatch.setattr(runner, "BATCH", 1)
    counts = {}
    read_summary, _, _ = runner.annotate("unused.bam", KINNEX8_SC3, 0.5, 1, counts=counts, molecules_out=Sink())
    assert received == list(range(1, 301))
    assert max(held) <= 4
    assert counts == {"hifi_reads_in_file": 300}
    assert read_summary["zm"].to_list() == [zm for zm in range(1, 301) if in_hash_sample(zm, 0.5)]


def _read_sreads_the_list_way(sreads, keep):
    """read_sreads' frame built the way it was built while it kept one Python list per column to the end."""
    cols = {"zm": [], "di": [], "qs": [], "qe": [], "dl": [], "dr": []}
    last_zm, di = None, -1
    for r in iter_bam(sreads, threads=1, keep=keep):
        zm = r.get_tag("zm")
        di = di + 1 if zm == last_zm else 0
        last_zm = zm
        for k, v in (("zm", zm), ("di", di), ("qs", r.get_tag("qs")), ("qe", r.get_tag("qe")),
                     ("dl", r.get_tag("dl")), ("dr", r.get_tag("dr"))):
            cols[k].append(v)
    return pl.DataFrame(cols, schema={"zm": pl.UInt32, "di": pl.Int16, "qs": pl.Int32, "qe": pl.Int32,
                                      "dl": pl.Int8, "dr": pl.Int8})


@pytest.mark.parametrize(("chunk", "fraction"), [(100_000, 1.0), (3, 1.0), (3, 0.5)])
def test_read_sreads_equals_the_list_built_reference(test_bams, monkeypatch, chunk, fraction):
    """Reading the S-reads in arrow chunks must not change a value, a dtype or the row order (di counts on across a
    chunk boundary): one chunk, chunks of 3, and a hash sample in chunks of 3."""
    _, sreads = test_bams
    monkeypatch.setattr(runner, "SREAD_CHUNK", chunk)
    got = runner.read_sreads(sreads, 1, keep=_sample(fraction))
    want = _read_sreads_the_list_way(sreads, _sample(fraction))
    assert got.height > 0
    assert got.schema == want.schema
    assert got.equals(want)


def test_read_sreads_keeps_rows_as_arrow_not_python_lists(monkeypatch):
    """read_sreads turns every SREAD_CHUNK S-reads into arrow, so the Python heap holds one chunk at a time.
    200,000 fake S-reads in chunks of 1,000: kept as Python lists to the end, they took a Python heap peak of
    28 MB (measured before the change; 0.2 MB after)."""
    def fake_iter_bam(path, threads=4, keep=None):
        for i in range(200_000):
            yield _FakeRecord({"zm": 1000 + i // 8, "qs": 300 * (i % 8), "qe": 300 * (i % 8) + 290, "dl": i % 8, "dr": i % 8 + 1})

    monkeypatch.setattr(runner, "iter_bam", fake_iter_bam)
    monkeypatch.setattr(runner, "SREAD_CHUNK", 1000, raising=False)
    sreads, peak = _python_heap_peak(lambda: runner.read_sreads("unused.bam", 1))
    assert sreads.height == 200_000
    assert sreads["di"].max() == 7
    assert peak < 5 * 2**20


def test_run_writes_reads_tsv_with_pattern_counts(test_bams, tmp_path):
    hifi, _ = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", "1", "--threads", "2", check=True)
    assert r.returncode == 0

    rows = _reads_tsv(out / "reads.tsv.gz")
    assert len(rows) == 6
    assert set(rows[0]) == {"zm", "read_len", "chain", "chain_flipped", "primary_pattern", "n_mas"}
    patterns = [row["primary_pattern"] for row in rows]
    assert patterns.count("complete") == 3
    assert patterns.count("mirror") == 3

    summary = json.loads((out / "summary.json").read_text())
    assert summary["hifi_reads_in_file"] == 6
    assert summary["hifi_reads"] == 6
    assert summary["fraction"] == 1.0
    assert summary["skera_version"] is None
    assert (out / "run.log").exists()


def test_fraction_zero_and_above_one_exit_2(test_bams, tmp_path):
    hifi, _ = test_bams
    for bad in ("0", "1.5"):
        r = _run("--hifi", str(hifi), "--out", str(tmp_path / bad), "--fraction", bad, "--threads", "2")
        assert r.returncode == 2
        assert r.stderr.startswith("kinnex-qc run: error:")


def test_write_molecules_keeps_the_default_fraction_for_the_summary(test_bams, tmp_path):
    """--write-molecules does not change the fraction: summary.json and reads.tsv.gz describe the default hash sample
    (0.1, zm 4 alone in the fixture), while molecules.bam holds the 30 usable molecules of all six reads."""
    hifi, sreads = test_bams
    assert {zm for zm in range(1, 7) if in_hash_sample(zm, 0.1)} == {4}
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--write-molecules", "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert "fraction set to 1" not in r.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["fraction"] == 0.1
    assert (summary["hifi_reads_in_file"], summary["hifi_reads"]) == (6, 1)
    assert [int(row["zm"]) for row in _reads_tsv(out / "reads.tsv.gz")] == [4]
    assert summary["molecules_written"] == 30
    assert sum(1 for _ in pysam.AlignmentFile(str(out / "molecules.bam"), "rb", check_sq=False)) == 30


def test_write_molecules_leaves_summary_and_reads_tsv_as_without_it(test_bams, tmp_path):
    """At a fraction below 1, with S-reads (so the skera comparison runs too), --write-molecules adds molecules_written
    to summary.json and changes nothing else in it or in reads.tsv.gz."""
    hifi, sreads = test_bams
    outs = {}
    for name, extra in (("with", ["--write-molecules"]), ("without", [])):
        out = tmp_path / name
        _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "0.5", "--threads", "2",
             *extra, check=True)
        with gzip.open(out / "reads.tsv.gz", "rt") as f:
            outs[name] = (json.loads((out / "summary.json").read_text()), f.read())
    assert outs["with"][0].pop("molecules_written") == 30
    assert outs["with"][0]["hifi_reads"] == 4
    assert outs["with"] == outs["without"]


def test_write_molecules_without_sreads_or_template_exits_2(test_bams, tmp_path):
    hifi, _ = test_bams
    r = _run("--hifi", str(hifi), "--out", str(tmp_path / "out"), "--write-molecules", "--threads", "2")
    assert r.returncode == 2
    assert r.stderr.startswith("kinnex-qc run: error:")


def test_write_molecules_with_template_missing_rg_exits_2(test_bams, tmp_path):
    hifi, _ = test_bams
    template = _sreads_bam_without_skera_pg(tmp_path / "no_rg.bam")
    r = _run("--hifi", str(hifi), "--out", str(tmp_path / "out"), "--write-molecules",
             "--template", str(template), "--threads", "2")
    assert r.returncode == 2
    assert r.stderr == f"kinnex-qc run: error: template {template} has no @RG read group\n"


def test_summary_records_skera_version_with_sreads(test_bams, tmp_path):
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "1", "--threads", "2",
              check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["skera_version"] == "26.1.0"


def test_run_emits_no_warnings(test_bams, tmp_path):
    """A run must not trigger polars' "pass a path, not a file object" UserWarning (reads.tsv.gz is
    written through a gzip handle, which write_csv() does not see as a path)."""
    hifi, _ = test_bams
    out = tmp_path / "out"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rc = cli.main(["run", "--hifi", str(hifi), "--out", str(out), "--fraction", "1", "--threads", "2"])
    assert rc == 0
    polars_warnings = [str(w.message) for w in caught if "polars" in str(w.message).lower()]
    assert polars_warnings == []


def test_hifi_reads_in_file_counts_whole_bam_at_any_fraction(test_bams, tmp_path):
    """hifi_reads_in_file must count every record in the HiFi BAM, not just the sampled ones, whether
    or not a sample was taken."""
    hifi, _ = test_bams
    n_records = sum(1 for _ in pysam.AlignmentFile(str(hifi), "rb", check_sq=False))
    for fraction in ("1", "0.5"):
        out = tmp_path / fraction
        r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", fraction, "--threads", "2", check=True)
        assert r.returncode == 0
        summary = json.loads((out / "summary.json").read_text())
        assert summary["hifi_reads_in_file"] == n_records


def test_bad_threads_exit_2(test_bams, tmp_path):
    hifi, _ = test_bams
    for bad in ("0", "-1"):
        r = _run("--hifi", str(hifi), "--out", str(tmp_path / bad), "--fraction", "1", "--threads", bad)
        assert r.returncode == 2
        assert r.stderr.startswith("kinnex-qc run: error:")


def test_negative_examples_exit_2(test_bams, tmp_path):
    hifi, _ = test_bams
    r = _run("--hifi", str(hifi), "--out", str(tmp_path / "out"), "--fraction", "1", "--examples", "-1")
    assert r.returncode == 2
    assert r.stderr == "kinnex-qc run: error: --examples must be >= 0: -1\n"


def test_adapters_naming_a_directory_exits_2(test_bams, tmp_path):
    hifi, _ = test_bams
    r = _run("--hifi", str(hifi), "--out", str(tmp_path / "out"), "--fraction", "1", "--threads", "2",
             "--adapters", str(tmp_path))
    assert r.returncode == 2
    assert r.stderr.startswith("kinnex-qc run: error:")


def test_out_naming_an_existing_file_exits_2(test_bams, tmp_path):
    hifi, _ = test_bams
    out = tmp_path / "out"
    out.write_text("not a directory")
    r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", "1", "--threads", "2")
    assert r.returncode == 2
    assert r.stderr.startswith("kinnex-qc run: error:")


def test_missing_hifi_exits_2(tmp_path):
    r = _run("--hifi", str(tmp_path / "nope.bam"), "--out", str(tmp_path / "out"), "--fraction", "1")
    assert r.returncode == 2
    assert r.stderr.startswith("kinnex-qc run: error:")


def test_bad_primers_exits_2(test_bams, tmp_path):
    hifi, _ = test_bams
    r = _run("--hifi", str(hifi), "--out", str(tmp_path / "out"), "--fraction", "1", "--primers", "bogus")
    assert r.returncode == 2
    assert r.stderr.startswith("kinnex-qc run: error:")


@_IGNORE_FORK_WARNING
def test_command_line_logs_the_argv_main_received_not_sys_argv(test_bams, tmp_path, monkeypatch):
    """runner.run must log the argv cli.main actually parsed, shell-quoted, not sys.argv (which differs
    for a library caller like the study's stage 02 that calls cli.main(argv) in-process)."""
    hifi, _ = test_bams
    out = tmp_path / "out with space"
    monkeypatch.setattr("sys.argv", ["pytest", "unrelated", "args"])
    rc = cli.main(["run", "--hifi", str(hifi), "--out", str(out), "--fraction", "1", "--threads", "1"])
    assert rc == 0
    log = (out / "run.log").read_text()
    [command_line] = [line for line in log.splitlines() if line.startswith("command:")]
    assert "unrelated" not in command_line
    assert f"--out '{out}'" in command_line


def test_hash_sample_annotates_exactly_the_sampled_reads(test_bams, tmp_path):
    hifi, _ = test_bams
    fraction = 0.5
    expected = {zm for zm in range(1, 7) if in_hash_sample(zm, fraction)}
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", str(fraction), "--threads", "2", check=True)
    assert r.returncode == 0
    rows = _reads_tsv(out / "reads.tsv.gz")
    assert {int(row["zm"]) for row in rows} == expected


def test_sreads_without_skera_pg_warns_and_records_null_version(test_bams, tmp_path):
    hifi, _ = test_bams
    sreads = _sreads_bam_without_skera_pg(tmp_path / "no_skera.bam")
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "1", "--threads", "2",
             check=True)
    assert r.returncode == 0
    assert f"warning: no @PG line names skera in {sreads}" in r.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["skera_version"] is None


@_IGNORE_FORK_WARNING
def test_summary_artifacts_and_junctions_match_the_molecules_frame(test_bams, tmp_path):
    hifi, _ = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", "1", "--threads", "2", check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())

    _, molecules, _ = runner.annotate(hifi, KINNEX8_SC3, 1.0, 2)
    tso_tso = molecules.filter(pl.col("end_class") == "tso_tso").height
    assert summary["artifacts"]["tso_tso_per_1000"] == 1000 * tso_tso / molecules.height
    assert summary["junctions"] == [
        {"letter": "B", "molecule": "R1-R1", "junctions": 1},
        {"letter": "H", "molecule": "TSO-TSO", "junctions": 2},
    ]


def test_summary_has_every_metrics_key_and_null_skera_pass_share_without_sreads(test_bams, tmp_path):
    hifi, _ = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", "1", "--threads", "2", check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())

    for key in ("patterns", "fractions", "artifacts", "mirror_junction_shares", "junctions",
                "junction_composition", "mirror_size_rule"):
        assert key in summary
    assert summary["junction_composition"]
    assert all(row["skera_pass_share"] is None for row in summary["junction_composition"])


def test_hash_sample_of_zero_reads_writes_null_fractions_not_a_crash(test_bams, tmp_path):
    """A fraction small enough to sample zero reads from the fixture must still finish with rc 0 and
    write summary.json and run.log, as it did before this task's read_fractions call (complete_fraction
    and mirror_fraction null rather than a mean-of-nothing TypeError)."""
    hifi, _ = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", "0.000001", "--threads", "1", check=True)
    assert r.returncode == 0, r.stderr
    assert (out / "run.log").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["hifi_reads"] == 0
    assert summary["fractions"] == {"complete_fraction": None, "mirror_fraction": None}


@_IGNORE_FORK_WARNING
def test_hash_sample_of_zero_reads_with_sreads_writes_null_budget_not_a_crash(test_bams, tmp_path):
    """The same zero-sampled-reads case as above, but with --sreads, must not crash in
    skera.budget_stats (budget_stats' per-column mean() of an empty frame is None, and
    float(None) used to raise TypeError; every budget value must come out null instead)."""
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "0.000001",
              "--threads", "1", check=True)
    assert r.returncode == 0, r.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["hifi_reads"] == 0
    assert summary["skera"]["budget"] == {
        "molecules": None, "unfilled_slots": None, "artifact_tso_tso": None, "artifact_r1_r1": None,
        "artifact_one_primer_or_none": None, "artifact_fused": None, "artifact": None, "usable": None,
        "usable_clean": None, "usable_not_clean": None,
    }


@_IGNORE_FORK_WARNING
def test_skera_summary_false_cut_and_mirror_arms(test_bams, tmp_path):
    """The test BAM's one false cut (around D, skera letter D, skipped adapter D) and its mirror
    arms (after CLEAN 1.0, before CLEAN 0.0) show up in summary.json["skera"]."""
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "1",
              "--threads", "2", check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())
    sk = summary["skera"]
    assert sk["complete_reads"] == 3
    assert sk["complete_passing_reads"] == 3
    assert sk["false_cuts"] == 1
    assert sk["false_cut_reads_share"] == pytest.approx(1 / 3)
    assert sk["false_cut_letters"] == {"D": 1}
    assert sk["skipped_adapter_letters"] == {"D": 1}
    assert sk["mirror_arms"]["after_clean_share"] == 1.0
    assert sk["mirror_arms"]["before_clean_share"] == 0.0


def test_skera_summary_absent_without_sreads(test_bams, tmp_path):
    hifi, _ = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", "1", "--threads", "2", check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())
    assert "skera" not in summary


@_IGNORE_FORK_WARNING
def test_reads_tsv_has_skera_columns_with_sreads(test_bams, tmp_path):
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "1",
              "--threads", "2", check=True)
    assert r.returncode == 0
    rows = {int(row["zm"]): row for row in _reads_tsv(out / "reads.tsv.gz")}
    assert set(rows[1]) == {"zm", "read_len", "chain", "chain_flipped", "primary_pattern", "n_mas",
                            "n_sreads", "passing", "false_cuts", "usable_not_clean"}
    assert int(rows[1]["n_sreads"]) == 8
    assert rows[1]["passing"] == "true"
    assert int(rows[1]["false_cuts"]) == 1
    assert int(rows[1]["usable_not_clean"]) == 2
    assert int(rows[6]["n_sreads"]) == 1
    assert rows[6]["passing"] == "false"


@_IGNORE_FORK_WARNING
def test_junction_composition_skera_pass_share_with_sreads(test_bams, tmp_path):
    """The three mirror reads (zm 4, 5: TSO-TSO at H; zm 6: R1-R1 at B) each have exactly one S-read,
    and each covers only the arm after the junction (dl, dr = 7, 8 for zm 4 and 5; 1, 0 for zm 6).
    None reaches both ends of the array (forward needs dl == 0, backward needs dl == order_len - 1
    == 8), so passing_from_labels calls all three failing and skera_pass_share must be exactly 0.0
    for both junction_molecules rows, not merely non-null."""
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "1",
              "--threads", "2", check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())
    rows = {row["junction_molecules"]: row for row in summary["junction_composition"]}
    assert rows["TSO-TSO only"]["reads"] == 2
    assert rows["TSO-TSO only"]["skera_pass_share"] == 0.0
    assert rows["R1-R1 only"]["reads"] == 1
    assert rows["R1-R1 only"]["skera_pass_share"] == 0.0


@_IGNORE_FORK_WARNING
def test_skera_budget_matches_hand_derived_numbers(test_bams, tmp_path):
    """Budget numbers worked out by hand from tests/make_test_bam.py's construction, not by calling
    skera.budget_stats a second time (which only checks the CLI agrees with itself, not with the
    fixture).

    zm 1, 2, 3: complete arrays, 8 molecules each, all end_class "proper" and not internal, so all 8
    are usable and none is an artifact. zm 2 and 3 have no false cut: their 8 S-reads match their 8
    molecules' bounds exactly, so all 8 are CLEAN. zm 1's false cut (around D) replaces the C-D and
    D-E S-reads: the C-D S-read now runs the FALSE_CUT_OFFSET (400 bp) into D-E as well as covering
    all of C-D, so it touches two molecules and C-D comes out FUSED rather than CLEAN; no single
    S-read then covers 80% of D-E, so D-E comes out PARTIAL. zm 1 is 6 CLEAN + 2 not-clean (1 FUSED,
    1 PARTIAL) of its 8 usable molecules.

    zm 4, 5 (TSO-TSO mirrors at H) and zm 6 (R1-R1 mirror at B): 3 molecules each, the middle one
    being the junction molecule itself (tso_tso for zm 4/5, r1_r1 for zm 6: not usable, one artifact
    each). Their single S-read covers exactly the arm after the junction (CLEAN); the arm before the
    junction has no S-read at all (NOT_WRITTEN). So 2 usable molecules each, 1 CLEAN / 1 not clean.

    molecules per read (1..6): 8, 8, 8, 3, 3, 3 -> mean 33 / 6
    unfilled_slots (order_len - 1 - molecules, order_len 9): 0, 0, 0, 5, 5, 5 -> mean 15 / 6
    artifact_tso_tso: 0, 0, 0, 1, 1, 0 -> mean 2 / 6
    artifact_r1_r1: 0, 0, 0, 0, 0, 1 -> mean 1 / 6
    artifact_one_primer_or_none, artifact_fused: 0 for every read -> mean 0.0 each
    artifact (the four artifact_* summed per read, then meaned): 0, 0, 0, 1, 1, 1 -> mean 3 / 6
    usable (proper, not internal): 8, 8, 8, 2, 2, 2 -> mean 30 / 6
    usable_clean: 6, 8, 8, 1, 1, 1 -> mean 25 / 6
    usable_not_clean: 2, 0, 0, 1, 1, 1 -> mean 5 / 6
    """
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "1",
              "--threads", "2", check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())
    budget = summary["skera"]["budget"]
    assert budget["molecules"] == pytest.approx(33 / 6)
    assert budget["unfilled_slots"] == pytest.approx(15 / 6)
    assert budget["artifact_tso_tso"] == pytest.approx(2 / 6)
    assert budget["artifact_r1_r1"] == pytest.approx(1 / 6)
    assert budget["artifact_one_primer_or_none"] == pytest.approx(0.0)
    assert budget["artifact_fused"] == pytest.approx(0.0)
    assert budget["artifact"] == pytest.approx(3 / 6)
    assert budget["usable"] == pytest.approx(30 / 6)
    assert budget["usable_clean"] == pytest.approx(25 / 6)
    assert budget["usable_not_clean"] == pytest.approx(5 / 6)


@_IGNORE_FORK_WARNING
def test_write_molecules_writes_molecules_bam_and_records_count(test_bams, tmp_path):
    """30 usable molecules in the fixture (hand-derived in test_skera_budget_matches_hand_derived_numbers:
    8 each for the three complete arrays, 2 each for the three mirrors)."""
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--write-molecules",
              "--threads", "2", check=True)
    assert r.returncode == 0, r.stderr

    mol_bam = out / "molecules.bam"
    assert mol_bam.exists()
    assert sum(1 for _ in pysam.AlignmentFile(str(mol_bam), "rb", check_sq=False)) == 30

    summary = json.loads((out / "summary.json").read_text())
    assert summary["molecules_written"] == 30
    assert ("molecules written: 30, for all 6 HiFi reads in the file; summary numbers come from the hash sample "
            "(fraction 0.1, 1 of 6 reads)") in (out / "run.log").read_text()


def test_write_molecules_at_fraction_1_logs_that_summary_numbers_come_from_all_reads(test_bams, tmp_path):
    """At fraction 1 there is no hash sample to name: the summary numbers come from every read."""
    hifi, sreads = test_bams
    out = tmp_path / "out"
    _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--write-molecules", "--fraction", "1",
         "--threads", "2", check=True)
    log = (out / "run.log").read_text()
    assert "molecules written: 30, for all 6 HiFi reads in the file; summary numbers come from all 6 reads\n" in log
    assert "hash sample" not in log


@_IGNORE_FORK_WARNING
def test_run_that_breaks_while_writing_molecules_leaves_no_molecules_bam(test_bams, tmp_path, monkeypatch):
    """A run that breaks in the annotation pass (a killed worker: BrokenProcessPool) after molecules were written leaves
    neither molecules.bam nor molecules.bam.part in DIR."""
    hifi, sreads = test_bams
    real_annotate = runner.annotate

    def breaks(*args, molecules_out=None, **kwargs):
        real_annotate(*args, molecules_out=molecules_out, **kwargs)
        assert molecules_out.written == 30
        raise BrokenProcessPool("a worker died")

    monkeypatch.setattr(runner, "annotate", breaks)
    out = tmp_path / "out"
    with pytest.raises(BrokenProcessPool):
        cli.main(["run", "--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--write-molecules",
                  "--threads", "1"])
    assert not (out / "molecules.bam").exists()
    assert not (out / "molecules.bam.part").exists()


@_IGNORE_FORK_WARNING
@pytest.mark.parametrize(("fraction", "extra"), [("1", []), ("0.5", []), ("0.5", ["--write-molecules"])])
def test_run_logs_the_progress_of_each_pass(test_bams, tmp_path, monkeypatch, capsys, fraction, extra):
    """run.log and stderr get a line every PROGRESS_EVERY HiFi records read in the annotation pass (2 here: lines at 2, 4
    and 6 of the fixture's 6 reads), and one at the start and end of the S-read pass and of the examples pass, in that
    order. Every read at fraction 1, a hash sample, and a hash sample with --write-molecules (every read annotated)."""
    hifi, sreads = test_bams
    monkeypatch.setattr(runner, "PROGRESS_EVERY", 2)
    out = tmp_path / "out"
    rc = cli.main(["run", "--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", fraction,
                   "--threads", "1", *extra])
    assert rc == 0
    log = (out / "run.log").read_text().splitlines()
    assert capsys.readouterr().err.splitlines() == log

    elapsed = r", elapsed \d+:\d\d:\d\d"
    zms = [r.get_tag("zm") for r in pysam.AlignmentFile(str(hifi), "rb", check_sq=False)]
    n_sreads = sum(in_hash_sample(r.get_tag("zm"), float(fraction))
                   for r in pysam.AlignmentFile(str(sreads), "rb", check_sq=False))
    want = [f"annotated: {n} records read, {sum(in_hash_sample(zm, float(fraction)) for zm in zms[:n])} in the sample"
            + elapsed for n in (2, 4, 6)]
    want += ["S-read pass: start" + elapsed, f"S-read pass: end, {n_sreads} S-reads of the checked reads" + elapsed,
             r"examples pass: start, \d+ example reads" + elapsed, "examples pass: end" + elapsed]
    got = [line for line in log if line.startswith(("annotated:", "S-read pass:", "examples pass:"))]
    assert len(got) == len(want)
    for line, pattern in zip(got, want):
        assert re.fullmatch(pattern, line), line


def test_threads_1_and_3_give_the_same_output(test_bams, tmp_path):
    hifi, _ = test_bams
    outs = []
    for threads in ("1", "3"):
        out = tmp_path / threads
        r = _run("--hifi", str(hifi), "--out", str(out), "--fraction", "1", "--threads", threads, check=True)
        assert r.returncode == 0
        # compare decompressed text, not raw gzip bytes: gzip's header embeds the write's mtime,
        # so two subprocess runs a second apart differ byte-for-byte even with identical content
        with gzip.open(out / "reads.tsv.gz", "rt") as f:
            outs.append(f.read())
    assert outs[0] == outs[1]
