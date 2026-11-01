"""Command line entry point: `kinnex-qc`."""
import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="kinnex-qc", description="QC of Kinnex arrays of single-cell cDNA")
    ap.add_argument("--version", action="version", version=f"kinnex-qc {__version__}")
    ap.add_subparsers(dest="command", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
