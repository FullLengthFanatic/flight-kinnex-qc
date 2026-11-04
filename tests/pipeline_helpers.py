"""Helpers for the study stage and analysis tests: the stage modules, the mini dataset, running a stage as the study
runs it, and comparing an analysis' output with its v10 golden files. Importing this module puts study/pipeline and
study/analyses on sys.path, so tests import the stage and analysis modules by name."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pysam

REPO = Path(__file__).resolve().parent.parent
PIPELINE = REPO / "study" / "pipeline"
ANALYSES = REPO / "study" / "analyses"
GOLDEN_INPUTS = Path(__file__).resolve().parent / "golden" / "v10_inputs"
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(ANALYSES))
sys.path.insert(0, str(REPO / "study"))

from compare_runs import compare_trees  # noqa: E402
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


def run_analysis(script: str, *args: str) -> subprocess.CompletedProcess:
    """Run one study analysis from the repository root; raise if it fails. Returns the finished process, with its
    stdout and stderr as text."""
    r = subprocess.run([sys.executable, str(ANALYSES / script), *args], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r


def golden_diffs(out: Path, golden: Path, scratch: Path, pattern: str = "*") -> list[str]:
    """compare_runs' differences between the golden files of one analysis whose names match pattern and the files of
    the same names in out (the analysis may write more files than were frozen). Copies both sides into scratch, so
    compare_trees sees the same file set on each; a golden file missing from out is a difference."""
    names = [p.relative_to(golden) for p in golden.rglob(pattern) if p.is_file()]
    for side, root in (("out", out), ("golden", golden)):
        for rel in names:
            if (root / rel).exists():
                (scratch / side / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / rel, scratch / side / rel)
    return compare_trees(scratch / "out", scratch / "golden")
