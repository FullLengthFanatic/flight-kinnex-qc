"""Tests for `kinnex-qc run`."""
import csv
import gzip
import json
import subprocess
import warnings

import polars as pl
import pysam
import pytest

from kinnex_qc import cli, runner, skera
from kinnex_qc.annotate import FAIL_PATTERNS, HIFI_PATTERNS
from kinnex_qc.patterns import KINNEX8_SC3
from kinnex_qc.sampling import in_hash_sample

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


def test_write_molecules_without_fraction_sets_fraction_to_one(test_bams, tmp_path):
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--write-molecules", "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert "fraction set to 1" in r.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["fraction"] == 1.0


def test_write_molecules_without_sreads_or_template_exits_2(test_bams, tmp_path):
    hifi, _ = test_bams
    r = _run("--hifi", str(hifi), "--out", str(tmp_path / "out"), "--write-molecules", "--threads", "2")
    assert r.returncode == 2
    assert r.stderr.startswith("kinnex-qc run: error:")


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
    for a library caller like Task 14's study that calls cli.main(argv) in-process)."""
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
    """Task 9 fix round 1: the same zero-sampled-reads case as above, but with --sreads, must not
    crash in skera.budget_stats (budget_stats' per-column mean() of an empty frame is None, and
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
    """Step 1 of Task 9: the test BAM's one false cut (around D, skera letter D, skipped adapter D)
    and its mirror arms (after CLEAN 1.0, before CLEAN 0.0) show up in summary.json["skera"]."""
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
def test_junction_composition_skera_pass_share_not_null_with_sreads(test_bams, tmp_path):
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "1",
              "--threads", "2", check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())
    assert all(row["skera_pass_share"] is not None for row in summary["junction_composition"])


@_IGNORE_FORK_WARNING
def test_skera_budget_matches_direct_library_call(test_bams, tmp_path):
    hifi, sreads = test_bams
    out = tmp_path / "out"
    r = _run("--hifi", str(hifi), "--sreads", str(sreads), "--out", str(out), "--fraction", "1",
              "--threads", "2", check=True)
    assert r.returncode == 0
    summary = json.loads((out / "summary.json").read_text())

    read_summary, molecules, features = runner.annotate(hifi, KINNEX8_SC3, 1.0, 2)
    sreads_df = runner.read_sreads(sreads, 2)
    cls = skera.budget_classes(molecules, sreads_df)
    expected = skera.budget_stats(read_summary, cls, order_len=len(KINNEX8_SC3.order))
    assert summary["skera"]["budget"] == expected


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
