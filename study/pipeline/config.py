"""Configuration and sample management for the study."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

EXPECTED_COLUMNS = (
    "sample",
    "cell",
    "depletion",
    "hifi_bam",
    "fail_bam",
    "segmented_bam",
    "non_passing_bam",
    "zmw_metrics",
    "skera_report",
    "fltnc_bam",
)


@dataclass(frozen=True)
class Sample:
    sample: str
    cell: str
    depletion: str
    hifi_bam: Path
    fail_bam: Path
    segmented_bam: Path
    non_passing_bam: Path
    zmw_metrics: Path
    skera_report: Path
    fltnc_bam: Path


def load_samples(path: Path | str) -> dict[str, Sample]:
    """Load samples from a TSV file.

    Expected columns: sample, cell, depletion, hifi_bam, fail_bam,
    segmented_bam, non_passing_bam, zmw_metrics, skera_report, fltnc_bam.
    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty samples file: {path}")

    header = lines[0].rstrip("\n").split("\t")
    missing = [c for c in EXPECTED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")

    samples = {}
    for line in lines[1:]:
        if not line.strip():
            continue

        parts = line.rstrip("\n").split("\t")
        row = dict(zip(header, parts, strict=True))

        sample = Sample(
            sample=row["sample"],
            cell=row["cell"],
            depletion=row["depletion"],
            hifi_bam=Path(row["hifi_bam"]),
            fail_bam=Path(row["fail_bam"]),
            segmented_bam=Path(row["segmented_bam"]),
            non_passing_bam=Path(row["non_passing_bam"]),
            zmw_metrics=Path(row["zmw_metrics"]),
            skera_report=Path(row["skera_report"]),
            fltnc_bam=Path(row["fltnc_bam"]),
        )
        samples[sample.sample] = sample

    return samples


def load_config(path: Path | str = "study/study.toml") -> dict:
    """Read study.toml and overlay study.local.toml from the same folder, if present."""
    path = Path(path)
    with open(path, "rb") as f:
        config = tomllib.load(f)

    local_path = path.with_name(f"{path.stem}.local{path.suffix}")
    if local_path.exists():
        with open(local_path, "rb") as f:
            config.update(tomllib.load(f))

    return config


def out_dir(out_root: Path | str, sample: str, fraction: float = 1) -> Path:
    """Output directory for a sample, created if needed: out_root/sample at fraction 1, and
    out_root/hash_<fraction>/sample for a hash sample of the cell's ZMWs (0 < fraction < 1)."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
    path = Path(out_root) / sample if fraction == 1 else Path(out_root) / f"hash_{fraction:g}" / sample
    path.mkdir(parents=True, exist_ok=True)
    return path
