"""Tests for study/run_study.sh on stub scripts: RUN_STUDY_SCRIPTS points the step table at a scratch copy of the study
folder layout whose scripts only record how they were called (and fail or hang when a test asks them to). Covers the
order and arguments of every step, the .ok markers and their skipping on a rerun, --from and --fresh, a failing stage
or hard gate stopping the run, a crashing skera gate only warning, INT and TERM stopping every chain, and the lock on
the out root."""
import fcntl
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from pipeline_helpers import REPO

RUN = REPO / "study" / "run_study.sh"
SAMPLES = ("BD144_ND", "BD144_HM", "BD144_kin")
COLUMNS = ("sample", "cell", "depletion", "hifi_bam", "fail_bam", "segmented_bam", "non_passing_bam", "zmw_metrics",
           "skera_report", "fltnc_bam")
STAGES = ("pipeline/s01_extract_segments.py", "pipeline/s02_annotate_reads.py", "pipeline/s03_build_reads.py",
          "pipeline/s05_fate.py", "pipeline/s06_concatenation_bias.py", "pipeline/s07_depletion.py", "pipeline/gates.py",
          "analyses/false_cuts_calibration.py", "analyses/false_cut_anatomy.py", "analyses/mirror_arms.py",
          "analyses/molecule_budget.py", "analyses/downstream_test.py", "analyses/discordant_reads.py",
          "analyses/chain_benchmark.py", "analyses/skera_whole_cell.py", "calibration/build_calibration_set.py",
          "calibration/skera_split.py", "calibration/head_version_check.py", "calibration/tso_end_validation.py",
          "argentag/sread_coords.py", "argentag/annotate_non_passing.py", "argentag/full_read_view.py",
          "argentag/independent_scan.py")
needs_gio = pytest.mark.skipif(shutil.which("gio") is None, reason="run_study.sh trashes markers with gio")

# Every stub appends "<script> <arguments>" to $STUB_LOG; it exits 1 when the line contains one of the ;-separated
# texts in $STUB_FAIL, and writes its pid to $STUB_PID and sleeps when the line contains $STUB_SLEEP.
STUB_PY = """import os
import sys
import time
from pathlib import Path

me = Path(__file__).resolve().relative_to(Path(os.environ["RUN_STUDY_SCRIPTS"]).resolve()).as_posix()
call = " ".join([me, *sys.argv[1:]])
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(call + "\\n")
if os.environ.get("STUB_SLEEP") and os.environ["STUB_SLEEP"] in call:
    Path(os.environ["STUB_PID"]).write_text(str(os.getpid()))
    time.sleep(120)
if any(t in call for t in os.environ.get("STUB_FAIL", "").split(";") if t):
    sys.exit(1)
"""
STUB_SH = """#!/bin/bash
call="pipeline/s04_lima_refine.sh $*"
echo "$call" >> "$STUB_LOG"
IFS=';' read -ra texts <<< "${STUB_FAIL:-}"
for t in "${texts[@]}"; do if [[ -n $t && $call == *"$t"* ]]; then exit 1; fi; done
exit 0
"""


class Study:
    """A stub script tree, a config and a samples table in tmp_path; run() runs run_study.sh on them."""

    def __init__(self, root: Path, fraction="1", extra=""):
        self.root = root
        self.scripts = root / "scripts"
        self.out = root / "out"
        self.log = root / "calls.log"
        self.samples = root / "samples.tsv"
        self.config = root / "cfg" / "study.toml"
        for rel in STAGES:
            (self.scripts / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.scripts / rel).write_text(STUB_PY)
        (self.scripts / "pipeline" / "s04_lima_refine.sh").write_text(STUB_SH)
        rows = ["\t".join(COLUMNS)] + ["\t".join([s] + [f"/data/{s}/{c}" for c in COLUMNS[1:]]) for s in SAMPLES]
        self.samples.write_text("\n".join(rows) + "\n")
        self.config.parent.mkdir()
        self.config.write_text(f'fraction = {fraction}\nfail_subsample = 100\nprocesses = 2\nthreads = 3\npbtools_env = "pb"\n'
                               f'samples = "{self.samples}"\nout_root = "{self.out}"\nreference_fasta = "/ref/genome.fa"\n'
                               f'skera120_env = "sk120"\nsamtools_env = "st113"\nargentag_dir = "<argentag>"\n{extra}')

    def env(self, **extra) -> dict:
        return {**os.environ, "RUN_STUDY_SCRIPTS": str(self.scripts), "STUB_LOG": str(self.log),
                "STUB_PID": str(self.root / "stub.pid"), **extra}

    def run(self, *args, **env) -> subprocess.CompletedProcess:
        """run_study.sh --config <config> args, from the repository root, after emptying the call log."""
        self.log.write_text("")
        return subprocess.run(["bash", str(RUN), "--config", str(self.config), *args], cwd=REPO, env=self.env(**env),
                              capture_output=True, text=True, timeout=300)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines()

    def markers(self) -> dict[str, str]:
        """<sample or all>/<step> -> text of every .ok marker under the out root's logs/."""
        return {f"{p.parent.name}/{p.name[:-3]}": p.read_text() for p in (self.out / "logs").glob("*/*.ok")}


@pytest.fixture
def study(tmp_path) -> Study:
    return Study(tmp_path)


def c(script: str, *args: str) -> str:
    return " ".join([script, *args])


def per_sample_calls(st: Study, s: str, mem: str) -> list[str]:
    """The calls of phases 0 to 2 for sample s at fraction 1, in order; mem stands for the memory budget."""
    common = ("--samples", str(st.samples), "--out-root", str(st.out), "--fraction", "1")
    gate = lambda g: c("pipeline/gates.py", "--sample", s, "--gate", g, *common, "--fail-subsample", "100")  # noqa: E731
    stage = lambda n, *a: c(f"pipeline/{n}.py", "--sample", s, *common, "--processes", "2", *a)  # noqa: E731
    return [gate("synthetic"), stage("s01_extract_segments"), stage("s02_annotate_reads", "--fail-subsample", "100"),
            stage("s03_build_reads"),
            c("pipeline/s04_lima_refine.sh", "--sample", s, *common, "--threads", "3", "--pbtools-env", "pb"),
            stage("s05_fate"), gate("skera"), gate("lima"), gate("adapter_index"),
            stage("s06_concatenation_bias", "--mem-budget-gb", mem), stage("s07_depletion")]


def phase4_calls(st: Study, run_dir: Path, argentag: Path | None = None, report: bool = False) -> list[str]:
    cal, a, nd = st.out / "calibration", run_dir / "analyses", run_dir / "BD144_ND"
    truth = cal / "nd_first200k_truth"
    calls = [
        c("calibration/build_calibration_set.py", "--samples", str(st.samples), "--out", str(cal), "--processes", "2",
          "--samtools", "", "--samtools-env", "st113"),
        c("calibration/skera_split.py", "--calibration-dir", str(cal), "--out", str(cal), "--skera-env", "sk120"),
        c("analyses/false_cuts_calibration.py", "--truth", str(truth), "--hifi", f"{cal}/nd_first200k.hifi.nomod.bam",
          "--out", f"{a}/skera_false_cuts"),
        *[c("analyses/false_cut_anatomy.py", "--sample-dir", str(run_dir / s), "--label", s, "--out", f"{a}/false_cut_anatomy")
          for s in SAMPLES],
        c("analyses/mirror_arms.py", "--truth", str(nd), "--segments", f"{nd}/segments.parquet", "--baseline-fraction", "0.01",
          "--label", "nd_hash_skera26", "--out", f"{a}/skera_mirror_arms"),
        c("analyses/mirror_arms.py", "--truth", str(truth), "--segments", f"{truth}/segments.parquet",
          "--label", "nd_cal_skera26", "--out", f"{a}/skera_mirror_arms"),
        c("analyses/mirror_arms.py", "--truth", str(truth), "--sreads", f"{cal}/nd_first200k.skera120.bam",
          "--non-passing", f"{cal}/nd_first200k.skera120.non_passing.bam",
          "--found-adapters", f"{cal}/nd_first200k.skera120.found_adapters.csv.gz",
          "--ligations", f"{cal}/nd_first200k.skera120.ligations.csv", "--label", "nd_cal_skera12", "--out", f"{a}/skera_mirror_arms"),
        *[c("analyses/molecule_budget.py", "--sample-dir", str(run_dir / s), "--label", s, "--out", f"{a}/molecule_budget")
          for s in SAMPLES],
        c("analyses/downstream_test.py", "--hifi", f"{cal}/nd_first200k.hifi.nomod.bam", "--template",
          f"{cal}/nd_first200k.skera120.bam", "--truth", str(truth), "--ref", "/ref/genome.fa", "--threads", "3",
          "--pbtools-env", "pb", "--minimap2", "", "--out", f"{a}/downstream_test"),
        c("analyses/discordant_reads.py", "--sample-dir", str(nd), "--out", f"{a}/discordant_reads"),
        c("analyses/chain_benchmark.py", "--sreads", f"{cal}/nd_first200k.skera120.bam", "--out", f"{a}/mirror_benchmark"),
        c("analyses/skera_whole_cell.py", "--samples", str(st.samples), "--out", f"{a}/skera_whole_cell"),
        c("calibration/head_version_check.py", "--calibration-dir", str(cal), "--out", str(cal)),
        c("calibration/tso_end_validation.py", "--calibration-dir", str(cal), "--out", str(cal)),
    ]
    if argentag:
        out = st.out / "argentag"
        calls += [c("argentag/sread_coords.py", "--argentag-dir", str(argentag), "--out", str(out)),
                  c("argentag/annotate_non_passing.py", "--argentag-dir", str(argentag), "--out", str(out), "--processes", "2"),
                  c("argentag/full_read_view.py", "--out", str(out)),
                  c("argentag/independent_scan.py", "--argentag-dir", str(argentag), "--out", str(out))]
    if report:
        calls += [c("report/compute.py", "--config", str(st.config), "--out-root", str(st.out)),
                  c("report/render.py", "--report-dir", str(run_dir / "report"))]
    return calls


def check_full_run(st: Study, calls: list[str], run_dir: Path, **phase4) -> None:
    """calls are every step of a whole run, each sample's chain in its order, the phases one after the other."""
    mem = re.search(r"--mem-budget-gb (\d+)", "\n".join(calls))
    assert mem, calls
    first_s01 = min(i for i, x in enumerate(calls) if "s01_extract_segments" in x)
    assert all("--gate synthetic" in x for x in calls[:first_s01]) and first_s01 == len(SAMPLES)
    for s in SAMPLES:
        expected = per_sample_calls(st, s, mem.group(1))
        assert [x for x in calls if f"--sample {s} " in x] == expected
    phase2 = [x for x in calls if re.search(r"s0[67]_\w+\.py --sample ", x)]
    assert phase2 == [x for s in SAMPLES for x in per_sample_calls(st, s, mem.group(1))[-2:]]   # one sample at a time
    assert calls.index(phase2[0]) > max(i for i, x in enumerate(calls) if "pipeline/gates.py" in x)
    common = ("--samples", str(st.samples), "--out-root", str(st.out), "--fraction", "1")
    after = calls[calls.index(phase2[-1]) + 1:]
    assert after == [c("pipeline/s06_concatenation_bias.py", "--all", *common), c("pipeline/s07_depletion.py", "--all", *common),
                     *phase4_calls(st, run_dir, **phase4)]


def test_whole_run_calls_every_step_in_order_with_its_arguments(study):
    r = study.run()
    assert r.returncode == 0, r.stderr
    check_full_run(study, study.calls(), study.out)
    assert "ArgenTag skipped: argentag_dir is not set" in r.stdout
    assert "report_compute skipped" in r.stdout and "report_render skipped" in r.stdout


def test_hash_fraction_argentag_and_report_steps(tmp_path):
    """Below fraction 1 the sample and analysis outputs sit under hash_<fraction>/; ArgenTag runs when argentag_dir holds
    both input BAMs, and the report steps run once their scripts exist."""
    arg = tmp_path / "argentag_in"
    (arg / "data").mkdir(parents=True)
    (arg / "data" / "segmented.bam").write_bytes(b"")
    (arg / "data" / "segmented.non_passing.bam").write_bytes(b"")
    st = Study(tmp_path / "st", fraction="0.18", extra=f'argentag_dir = "{arg}"\n')
    st.config.write_text(st.config.read_text().replace('argentag_dir = "<argentag>"\n', ""))
    (st.scripts / "report").mkdir()
    for name in ("compute.py", "render.py"):
        (st.scripts / "report" / name).write_text(STUB_PY)
    r = st.run()
    assert r.returncode == 0, r.stderr
    calls = st.calls()
    assert c("pipeline/s01_extract_segments.py", "--sample", "BD144_ND", "--samples", str(st.samples), "--out-root", str(st.out),
             "--fraction", "0.18", "--processes", "2") in calls
    run_dir = st.out / "hash_0.18"
    assert calls[-len(phase4_calls(st, run_dir, arg, True)):] == phase4_calls(st, run_dir, arg, True)
    assert "skipped" not in r.stdout


def test_argentag_skipped_when_an_input_is_missing(tmp_path):
    arg = tmp_path / "argentag_in"
    (arg / "data").mkdir(parents=True)
    (arg / "data" / "segmented.bam").write_bytes(b"")
    st = Study(tmp_path / "st")
    st.config.write_text(st.config.read_text().replace('"<argentag>"', f'"{arg}"'))
    r = st.run()
    assert r.returncode == 0, r.stderr
    assert not any(x.startswith("argentag/") for x in st.calls())
    assert f"ArgenTag skipped: {arg}/data/segmented.non_passing.bam does not exist" in r.stdout


def test_report_steps_skipped_when_argentag_is_skipped(tmp_path):
    """The Note needs the ArgenTag check: with ArgenTag skipped, the report steps are skipped too, with the reason logged,
    even though their scripts exist."""
    st = Study(tmp_path / "st")
    (st.scripts / "report").mkdir()
    for name in ("compute.py", "render.py"):
        (st.scripts / "report" / name).write_text(STUB_PY)
    r = st.run()
    assert r.returncode == 0, r.stderr
    assert not any(x.startswith(("argentag/", "report/")) for x in st.calls())
    for step in ("report_compute", "report_render"):
        assert re.search(rf"{step} skipped: the Note needs the ArgenTag check, which was skipped \(argentag_dir is not set\)", r.stdout)


def test_markers_written_and_steps_skipped_on_rerun(study):
    assert study.run().returncode == 0
    m = study.markers()
    steps = ("gate_synthetic", "s01_extract_segments", "s02_annotate_reads", "s03_build_reads", "s04_lima_refine", "s05_fate",
             "gate_skera", "gate_lima", "gate_adapter_index", "s06_concatenation_bias", "s07_depletion")
    assert {f"{s}/{x}" for s in SAMPLES for x in steps} <= set(m)
    assert {"all/s06_concatenation_bias", "all/s07_depletion", "all/build_calibration_set", "all/mirror_arms.nd_cal_skera12",
            "all/false_cut_anatomy.BD144_kin", "all/tso_end_validation"} <= set(m)
    assert len(m) == len(SAMPLES) * len(steps) + 2 + len(phase4_calls(study, study.out))
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \S+ \d+( fraction 1)?\n", t) for t in m.values()), m
    assert (study.out / "logs" / "BD144_ND" / "s01_extract_segments.log").exists()
    r = study.run()
    assert r.returncode == 0, r.stderr
    assert study.calls() == []
    assert "BD144_ND s01_extract_segments skip" in r.stdout
    assert study.markers() == m


SHARED_STEPS = ("build_calibration_set", "skera_split", "head_version_check", "tso_end_validation")


def test_a_new_fraction_runs_again_every_step_whose_outputs_sit_under_the_fraction_folder(study):
    """The markers sit in <out root>/logs whatever the fraction, so a marker of a step whose outputs depend on the
    fraction records it: a run at fraction 0.18 after one at fraction 1 into the same out root runs the per-sample,
    cross-sample and analysis steps again, and skips the calibration steps, whose outputs are shared."""
    assert study.run().returncode == 0
    m = study.markers()
    assert {k for k, t in m.items() if not t.endswith(" fraction 1\n")} == {f"all/{x}" for x in SHARED_STEPS}
    study.config.write_text(study.config.read_text().replace("fraction = 1\n", "fraction = 0.18\n"))
    r = study.run()
    assert r.returncode == 0, r.stderr
    calls = study.calls()
    at = lambda xs: [re.sub(r"--fraction 1\b", "--fraction 0.18", x) for x in xs]  # noqa: E731
    mem = re.search(r"--mem-budget-gb (\d+)", "\n".join(calls))
    assert mem, calls
    for s in SAMPLES:
        assert [x for x in calls if f"--sample {s} " in x] == at(per_sample_calls(study, s, mem.group(1)))
    common = ("--samples", str(study.samples), "--out-root", str(study.out), "--fraction", "0.18")
    after = calls[calls.index(at(per_sample_calls(study, SAMPLES[-1], mem.group(1)))[-1]) + 1:]
    assert after == [c("pipeline/s06_concatenation_bias.py", "--all", *common), c("pipeline/s07_depletion.py", "--all", *common),
                     *[x for x in phase4_calls(study, study.out / "hash_0.18") if not x.startswith("calibration/")]]
    assert "BD144_ND s01_extract_segments: the .ok marker is not from fraction 0.18" in r.stdout
    assert all(f"all {x} skip" in r.stdout for x in SHARED_STEPS)
    m = study.markers()
    assert {k for k, t in m.items() if not t.endswith(" fraction 0.18\n")} == {f"all/{x}" for x in SHARED_STEPS}
    r = study.run()
    assert r.returncode == 0, r.stderr
    assert study.calls() == []


def test_argentag_markers_are_shared_by_every_fraction(tmp_path):
    arg = tmp_path / "argentag_in"
    (arg / "data").mkdir(parents=True)
    (arg / "data" / "segmented.bam").write_bytes(b"")
    (arg / "data" / "segmented.non_passing.bam").write_bytes(b"")
    st = Study(tmp_path / "st")
    st.config.write_text(st.config.read_text().replace('"<argentag>"', f'"{arg}"'))
    assert st.run().returncode == 0
    assert any(x.startswith("argentag/") for x in st.calls())
    st.config.write_text(st.config.read_text().replace("fraction = 1\n", "fraction = 0.18\n"))
    r = st.run()
    assert r.returncode == 0, r.stderr
    assert not any(x.startswith(("argentag/", "calibration/")) for x in st.calls())
    assert any(x.startswith("analyses/") for x in st.calls())


@needs_gio
def test_from_trashes_the_markers_of_that_step_and_every_later_one(study):
    assert study.run().returncode == 0
    before = study.markers()
    r = study.run("--from", "s06")
    assert r.returncode == 0, r.stderr
    calls = study.calls()
    assert calls[0].startswith("pipeline/s06_concatenation_bias.py --sample BD144_ND ")
    assert not any(re.search(r"s0[1-5]_|gates\.py", x) for x in calls)
    assert len(calls) == 2 * len(SAMPLES) + 2 + len(phase4_calls(study, study.out))
    assert f"--from s06: trashed {len(calls)} .ok markers" in r.stdout
    after = study.markers()
    assert after.keys() == before.keys()
    assert after["BD144_HM/s05_fate"] == before["BD144_HM/s05_fate"]
    r = study.run("--from", "mirror_arms")
    assert r.returncode == 0, r.stderr
    phase4 = phase4_calls(study, study.out)
    assert study.calls() == phase4[next(i for i, x in enumerate(phase4) if x.startswith("analyses/mirror_arms.py")):]


@needs_gio
def test_fresh_trashes_every_marker(study):
    assert study.run().returncode == 0
    n = len(study.calls())
    r = study.run("--fresh")
    assert r.returncode == 0, r.stderr
    assert len(study.calls()) == n
    assert f"--fresh: trashed {len(study.markers())} .ok markers" in r.stdout


def test_from_names_no_step(study):
    r = study.run("--from", "s99")
    assert r.returncode == 2
    assert "no step starts with 's99'" in r.stderr


def test_failing_hard_gate_stops_the_run_after_the_other_gates_of_its_sample(study):
    r = study.run(STUB_FAIL="--sample BD144_HM --gate lima")
    assert r.returncode == 1
    calls = study.calls()
    assert any("--sample BD144_HM --gate adapter_index" in x for x in calls)
    assert not any("s06_" in x or x.startswith(("analyses/", "calibration/")) for x in calls)
    assert "FAILED: BD144_HM gate_lima" in r.stderr and "STOPPED" in r.stderr
    m = study.markers()
    assert "BD144_HM/gate_adapter_index" in m and "BD144_HM/gate_lima" not in m


def test_failing_stage_stops_its_chain_and_the_run(study):
    r = study.run(STUB_FAIL="s03_build_reads.py --sample BD144_ND")
    assert r.returncode == 1
    calls = study.calls()
    assert not any("--sample BD144_ND " in x and ("s04_" in x or "--gate skera" in x) for x in calls)
    assert not any("s06_" in x for x in calls)
    assert "FAILED: BD144_ND s03_build_reads" in r.stderr


def test_failing_analysis_stops_the_run(study):
    r = study.run(STUB_FAIL="analyses/downstream_test.py")
    assert r.returncode == 1
    assert study.calls()[-1].startswith("analyses/downstream_test.py ")
    assert "FAILED: all downstream_test" in r.stderr


@needs_gio
def test_skera_gate_crash_only_warns_and_trashes_a_stale_gate_file(study):
    stale = study.out / "BD144_ND" / "gates" / "gate_skera.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    r = study.run(STUB_FAIL="--gate skera")
    assert r.returncode == 0, r.stderr
    assert not stale.exists()
    assert all(f"WARNING: {s} gate_skera crashed" in r.stderr for s in SAMPLES)
    assert not any(k.endswith("/gate_skera") for k in study.markers())
    check_full_run(study, study.calls(), study.out)


def _gone(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1].startswith("Z")
    except FileNotFoundError:
        return True


@pytest.mark.parametrize("sig, code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_signal_stops_every_chain(study, sig, code):
    study.log.write_text("")
    pid_file = study.root / "stub.pid"
    p = subprocess.Popen(["bash", str(RUN), "--config", str(study.config)], cwd=REPO, start_new_session=True,
                         env=study.env(STUB_SLEEP="s01_extract_segments.py --sample BD144_ND"),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(600):
            if pid_file.exists() and pid_file.read_text():
                break
            time.sleep(0.05)
        stub = int(pid_file.read_text())
        assert not _gone(stub)
        p.send_signal(sig)
        _, err = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == code
    assert "STOPPED: run_study.sh got SIG" in err
    for _ in range(100):
        if _gone(stub):
            break
        time.sleep(0.1)
    assert _gone(stub)
    assert not any("s02_annotate_reads.py --sample BD144_ND" in x for x in study.calls())


def test_second_run_on_the_same_out_root_exits_3(study):
    lock = study.out / "logs" / ".run_study.lock"
    lock.parent.mkdir(parents=True)
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = study.run()
    assert r.returncode == 3
    assert str(lock) in r.stderr
    assert study.calls() == []


def test_placeholder_out_root_needs_out(study):
    study.config.write_text(study.config.read_text().replace(f'out_root = "{study.out}"', 'out_root = "<out>"'))
    r = study.run()
    assert r.returncode == 2
    assert "out_root" in r.stderr
    other = study.root / "other"
    r = study.run("--out", str(other))
    assert r.returncode == 0, r.stderr
    assert (other / "logs" / "all" / "s07_depletion.ok").exists()
