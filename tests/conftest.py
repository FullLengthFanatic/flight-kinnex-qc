import os
import tomllib
from pathlib import Path

import pytest


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
