"""Helpers for the study stage tests: the stage modules, the mini dataset and running a stage as the study runs it.
Importing this module puts study/pipeline on sys.path, so tests import the stage modules by name."""
import json
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


def run_cross_sample(script: str, out_root: Path, samples: Path, *extra: str) -> None:
    """Run the cross-sample mode (--all) of stage 06 or 07, from the repository root; raise if it fails."""
    subprocess.run([sys.executable, str(PIPELINE / script), "--all", "--samples", str(samples), "--out-root", str(out_root), *extra],
                   cwd=REPO, check=True)


def key_structure(x):
    """The keys of a JSON value, in order and at every depth; a list by the distinct structures of its items."""
    if isinstance(x, dict):
        return [(k, key_structure(v)) for k, v in x.items()]
    if isinstance(x, list):
        return sorted({json.dumps(key_structure(v)) for v in x})
    return None
