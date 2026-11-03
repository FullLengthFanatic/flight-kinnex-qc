"""Helpers for the study stage tests: the stage modules, the mini dataset and running a stage as the study runs it.
Importing this module puts study/pipeline on sys.path, so tests import the stage modules by name."""
import subprocess
import sys
from pathlib import Path

import pysam

REPO = Path(__file__).resolve().parent.parent
PIPELINE = REPO / "study" / "pipeline"
sys.path.insert(0, str(PIPELINE))

from config import load_samples  # noqa: E402


def mini_samples(mini: Path) -> dict:
    return load_samples(mini / "samples_mini.tsv")


def bam_records(path) -> list[pysam.AlignedSegment]:
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return list(bam)


def run_stage(script: str, sample: str, out_root: Path, samples: Path, *extra: str) -> None:
    """Run one study stage on one sample, from the repository root, with --samples and --out-root given; raise if it
    fails. Python stages get --processes 2 unless extra sets it."""
    if script.endswith(".sh"):
        cmd = ["bash", str(PIPELINE / script)]
    else:
        cmd = [sys.executable, str(PIPELINE / script)]
        if "--processes" not in extra:
            cmd += ["--processes", "2"]
    cmd += ["--sample", sample, "--samples", str(samples), "--out-root", str(out_root), *extra]
    subprocess.run(cmd, cwd=REPO, check=True)
