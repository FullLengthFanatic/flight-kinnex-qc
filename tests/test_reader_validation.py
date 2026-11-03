"""validate_reader.py against the real BD144 kinnex-qc run output (task 13's controller runs)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "study"))

from validate_reader import SAMPLES, main  # noqa: E402


@pytest.mark.real
def test_validate_reader_matches_v10(local):
    validation_dir = local["validation_dir"]
    if not any((validation_dir / sample / "summary.json").is_file() for sample in SAMPLES):
        pytest.skip(f"no kinnex-qc run output under {validation_dir}; run the three BD144 validation runs first")
    assert main([str(validation_dir)]) == 0
