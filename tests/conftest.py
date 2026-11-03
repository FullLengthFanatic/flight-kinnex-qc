import os
import shutil
import tomllib
from pathlib import Path

import pytest

import make_test_bam


@pytest.fixture(scope="session")
def test_bams(tmp_path_factory) -> tuple[Path, Path]:
    """(hifi_bam, sreads_bam) for the synthetic test array: see make_test_bam.build. Written once per
    session into tmp_path_factory, never into the repo."""
    return make_test_bam.build(tmp_path_factory.mktemp("kinnex_qc_test_bam"))


@pytest.fixture(scope="session")
def local() -> dict:
    """This machine's data paths, from the git-ignored TOML named by KINNEX_QC_LOCAL. Every value is
    wrapped in Path, including non-path ones (pbtools_env: the conda environment lima and isoseq run
    in; str() of it is the plain name). validation_dir: the three `kinnex-qc run` BD144 validation
    runs' output, <validation_dir>/<sample>/ holding summary.json, reads.tsv.gz, run.log and
    report.html (study/validate_reader.py checks it against the v10 study numbers)."""
    path = os.environ.get("KINNEX_QC_LOCAL")
    if not path:
        pytest.skip("KINNEX_QC_LOCAL not set")
    return {k: Path(v) for k, v in tomllib.loads(Path(path).read_text()).items()}


@pytest.fixture(scope="session")
def mini(request) -> Path:
    """The mini dataset study/make_mini.py builds: $KINNEX_QC_MINI, else mini_dir in the TOML named by
    KINNEX_QC_LOCAL. Holds <sample>/ with the plain-file inputs and samples_mini.tsv."""
    path = os.environ.get("KINNEX_QC_MINI")
    path = Path(path) if path else request.getfixturevalue("local")["mini_dir"]
    if not (path / "samples_mini.tsv").exists():
        pytest.fail(f"no mini dataset in {path}: build it with study/make_mini.py")
    return path


@pytest.fixture(scope="session")
def nd_out(mini, tmp_path_factory) -> Path:
    """Stages 01 to 03 run once on the mini BD144_ND at fraction 1, 1 in 10 fail reads annotated; the sample's
    output directory."""
    from pipeline_helpers import run_stage
    out = tmp_path_factory.mktemp("nd")
    for stage, extra in (("s01_extract_segments.py", ()), ("s02_annotate_reads.py", ("--fail-subsample", "10")),
                         ("s03_build_reads.py", ())):
        run_stage(stage, "BD144_ND", out, mini / "samples_mini.tsv", "--fraction", "1", *extra)
    return out / "BD144_ND"


@pytest.fixture(scope="session")
def mini_study(mini, tmp_path_factory) -> Path:
    """Stages 01 to 07 run once on the three mini samples at fraction 1, 1 in 10 fail reads annotated, stage 06 with
    10 null replicates and 100,000 simulated arrays; the output root. The gate stage is not part of this fixture: each
    sample's frozen gate_skera.json (tests/golden) is put in place before stage 07, so q2_summary.json carries
    skera_agreement, as v10's did."""
    from pipeline_helpers import run_stage
    out = tmp_path_factory.mktemp("study")
    samples = mini / "samples_mini.tsv"
    golden = Path(__file__).parent / "golden" / "v10_inputs" / "hash_0.18"
    for s in ("BD144_ND", "BD144_HM", "BD144_kin"):
        for stage, extra in (("s01_extract_segments.py", ()), ("s02_annotate_reads.py", ("--fail-subsample", "10")),
                             ("s03_build_reads.py", ()), ("s04_lima_refine.sh", ()), ("s05_fate.py", ()),
                             ("s06_concatenation_bias.py", ("--n-null", "10", "--n-sim", "100000"))):
            run_stage(stage, s, out, samples, "--fraction", "1", *extra)
        (out / s / "gates").mkdir()
        shutil.copy(golden / s / "gates" / "gate_skera.json", out / s / "gates")
        run_stage("s07_depletion.py", s, out, samples, "--fraction", "1")
    return out


def pytest_collection_modifyitems(config, items):
    if os.environ.get("KINNEX_QC_REAL") == "1":
        return
    skip = pytest.mark.skip(reason="needs local BD144/ArgenTag data; set KINNEX_QC_REAL=1")
    for item in items:
        if "real" in item.keywords:
            item.add_marker(skip)
