"""Command line entry point: `kinnex-qc`."""
import argparse

from . import __version__, runner


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="kinnex-qc", description="QC of Kinnex arrays of single-cell cDNA")
    ap.add_argument("--version", action="version", version=f"kinnex-qc {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", description="Annotate a HiFi BAM: MAS adapter chains, mirror arrays, "
                                              "and (with --sreads) skera's cutting.")
    run.add_argument("--hifi", required=True, help="HiFi (CCS) BAM")
    run.add_argument("--sreads", help="skera's S-read BAM")
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--adapters", help="adapter FASTA (default: the Kinnex 8-fold adapters)")
    run.add_argument("--primers", default="sc3", help="primer preset (sc3) or 'TSO,READ1' (default: sc3)")
    run.add_argument("--fraction", type=float, default=None,
                      help="share of ZMWs to annotate, in (0, 1] (default: 0.1, or 1.0 with --write-molecules)")
    run.add_argument("--threads", type=int, default=4, help="worker processes (default: 4)")
    run.add_argument("--write-molecules", action="store_true", help="write a per-molecule table")
    run.add_argument("--template", help="BAM whose header to copy for --write-molecules (default: --sreads)")
    run.add_argument("--examples", type=int, default=2, help="example reads per pattern to keep (default: 2)")

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return runner.run(args)
    return 0
