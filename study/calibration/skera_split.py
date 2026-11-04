#!/usr/bin/env python3
"""skera 1.2.0 on the calibration set, the version behind the public ArgenTag S-reads, as the report ran it:

    skera split -j 16 nd_first200k.hifi.nomod.bam mas8_adapters.fasta nd_first200k.skera120.bam

nd_first200k.hifi.nomod.bam comes from build_calibration_set.py; mas8_adapters.fasta (next to this script) holds the
Kinnex 8-fold adapters, the same sequences as kinnex_qc.patterns.MAS. skera writes nd_first200k.skera120.bam and its
companion files into --out: non_passing.bam, summary.csv, ligations.csv and read_lengths.csv (documented at
https://skera.how/output.html), and found_adapters.csv.gz and summary.json (observed with skera 1.2.0). skera runs in the conda
environment --skera-env (bioconda pbskera=1.2.0, envs/skera120.yml); its version goes to stderr, the run log.
Usage, from the repository root:
  study/calibration/skera_split.py [--calibration-dir DIR] [--out DIR] [--skera-env NAME|PATH] [--threads 16]
--calibration-dir and --skera-env default to calibration_dir and skera120_env of study/study.toml and
study.local.toml; --out defaults to the calibration folder.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import load_config  # noqa: E402

ADAPTERS = Path(__file__).resolve().parent / "mas8_adapters.fasta"


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration-dir", default=config.get("calibration_dir"), help="calibration folder (build_calibration_set.py)")
    ap.add_argument("--out", help="output directory (default: the calibration folder)")
    ap.add_argument("--skera-env", default=config.get("skera120_env"), help="conda environment with skera 1.2.0 (a name, or a path)")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(argv)
    if not args.calibration_dir or args.calibration_dir.startswith("<"):
        ap.error("no calibration folder: give --calibration-dir or set calibration_dir in study/study.local.toml")
    cal = Path(args.calibration_dir)
    out = Path(args.out) if args.out else cal
    out.mkdir(parents=True, exist_ok=True)
    run = ["conda", "run", "--no-capture-output", "-p" if "/" in args.skera_env else "-n", args.skera_env]
    version = subprocess.run(run + ["skera", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0].strip()
    print(version, file=sys.stderr)
    subprocess.run(run + ["skera", "split", "-j", str(args.threads), str(cal / "nd_first200k.hifi.nomod.bam"), str(ADAPTERS),
                          str(out / "nd_first200k.skera120.bam")], check=True)


if __name__ == "__main__":
    main()
