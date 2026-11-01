import os
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
    """This machine's data paths, from the git-ignored TOML named by KINNEX_QC_LOCAL."""
    path = os.environ.get("KINNEX_QC_LOCAL")
    if not path:
        pytest.skip("KINNEX_QC_LOCAL not set")
    return {k: Path(v) for k, v in tomllib.loads(Path(path).read_text()).items()}


def pytest_collection_modifyitems(config, items):
    if os.environ.get("KINNEX_QC_REAL") == "1":
        return
    skip = pytest.mark.skip(reason="needs local BD144/ArgenTag data; set KINNEX_QC_REAL=1")
    for item in items:
        if "real" in item.keywords:
            item.add_marker(skip)
