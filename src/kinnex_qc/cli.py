"""Command line entry point: `kinnex-qc`. Option parsing, the `run` rule checks and their one-line
errors and exit codes live here; runner.py takes the already-validated values and writes the output,
so it stays importable as a library (the study's stage 02) without argparse."""
import argparse
import sys
from pathlib import Path

import pysam

from . import __version__, runner
from .patterns import chemistry_from


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
    run.add_argument("--fraction", type=float, default=0.1,
                      help="share of ZMWs the summary and report describe, in (0, 1] (default: 0.1)")
    run.add_argument("--threads", type=int, default=4, help="worker processes (default: 4)")
    run.add_argument("--write-molecules", action="store_true",
                      help="write DIR/molecules.bam: one unaligned record per usable molecule of every read in the BAM "
                           "(not only the sampled ones), ready for lima --isoseq")
    run.add_argument("--template", help="BAM whose header to copy for --write-molecules (default: --sreads)")
    run.add_argument("--examples", type=int, default=2,
                      help="example reads per category shown in DIR/report.html (default: 2; 0 for none)")

    return ap


def _fail(msg: str) -> int:
    print(f"kinnex-qc run: error: {msg}", file=sys.stderr)
    return 2


def _run_cmd(args: argparse.Namespace, argv: list[str]) -> int:
    """Validate `kinnex-qc run`'s options and, if they are all good, call runner.run to do the work.
    Returns the process exit code."""
    if args.threads < 1:
        return _fail(f"--threads must be >= 1: {args.threads}")
    if args.examples < 0:
        return _fail(f"--examples must be >= 0: {args.examples}")

    hifi = Path(args.hifi)
    if not hifi.is_file():
        return _fail(f"--hifi: no such file: {hifi}")
    sreads = Path(args.sreads) if args.sreads else None
    if sreads is not None and not sreads.is_file():
        return _fail(f"--sreads: no such file: {sreads}")
    if args.adapters is not None and not Path(args.adapters).is_file():
        return _fail(f"--adapters: no such file: {args.adapters}")
    if args.template is not None and not Path(args.template).is_file():
        return _fail(f"--template: no such file: {args.template}")
    template = Path(args.template) if args.template else sreads

    if args.write_molecules and sreads is None and template is None:
        return _fail("--write-molecules needs --sreads or --template")
    if args.write_molecules:
        with pysam.AlignmentFile(str(template), "rb", check_sq=False) as t:
            if not t.header.to_dict().get("RG"):
                return _fail(f"template {template} has no @RG read group")

    out = Path(args.out)
    if out.exists() and not out.is_dir():
        return _fail(f"--out: exists and is not a directory: {out}")

    fraction = args.fraction
    if not (0 < fraction <= 1):
        return _fail(f"--fraction must be in (0, 1]: {fraction}")

    try:
        chem = chemistry_from(args.adapters, args.primers)
    except ValueError as e:
        return _fail(str(e))

    return runner.run(hifi, sreads, out, chem, fraction, args.threads, args.write_molecules, template,
                       args.examples, adapters=args.adapters, primers=args.primers,
                       hifi_arg=args.hifi, sreads_arg=args.sreads, template_arg=args.template, argv=argv)


def main(argv: list[str] | None = None) -> int:
    used_argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run_cmd(args, used_argv)
    return 0
