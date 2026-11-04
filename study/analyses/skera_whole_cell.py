#!/usr/bin/env python3
"""skera's own whole-cell Read Segmentation reports, copied into the output tree.

Not an analysis of ours: the report compares its hash sample with the whole cell, and the whole-cell numbers are skera's
(read_segmentation.report.json, the skera_report column of the samples table). Writes <sample>.json into --out for every
sample of the table, a byte-for-byte copy.
Usage, from the repository root:
  study/analyses/skera_whole_cell.py --out <dir> [--samples PATH]
Defaults come from study/study.toml.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config, load_samples  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=config["samples"], help="samples table")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, sample in load_samples(args.samples).items():
        shutil.copyfile(sample.skera_report, out / f"{name}.json")


if __name__ == "__main__":
    main()
